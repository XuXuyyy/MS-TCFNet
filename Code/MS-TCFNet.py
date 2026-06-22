# -*- coding: utf-8 -*-
"""
MS-TCFNet .py

Final EHDI reconstruction script.

Purpose:
1. Read restored 7-band monthly Stack_YYYY_MM.tif sequences: SPI, LST, NDVI, NDII, MSI, TCW, and MNDWI.
2. Train and evaluate MS-TCFNet with a 6-month temporal window.
3. Use 30 m supervisory SPEI, structural regularization, and 1 km aggregation consistency jointly.
4. Run five-fold forward-chaining time-series cross-validation.
5. Save checkpoints, metrics, standardization parameters, sampled validation points, and optional EHDI rasters.

Final configuration:
- Model: ConvLSTM + Temporal Transformer + FiLM + U-Net decoder.
- Hyperopt search utility: ConvLSTM hidden channels, U-Net base channels,
  Transformer embedding dimension, attention heads, and encoder layers.
- Loss: L0 pixel Huber + L1 structural loss + L2 aggregation-consistency loss.
- Product: 30 m EHDI.

2026.5.27 Xu
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("VSI_CACHE", "FALSE")

import gc
import json
import math
import random
import warnings
import hashlib
from typing import List
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from tqdm import tqdm

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
except ImportError:
    fmin = tpe = hp = Trials = STATUS_OK = STATUS_FAIL = None

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

warnings.filterwarnings("ignore")


# =========================================================
# 0) Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_exists(path: str, what: str = "file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")


def _transform_close(t1, t2, eps=1e-6):
    return all(abs(a - b) <= eps for a, b in zip(tuple(t1), tuple(t2)))


def add_months(yyyymm: str, k: int) -> str:
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    total = y * 12 + (m - 1) + k
    yy = total // 12
    mm = total % 12 + 1
    return f"{yy:04d}{mm:02d}"


def list_months(start: str, end: str) -> List[str]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def build_tscv_month_splits(all_months: List[str], n_splits: int = 5, test_size: int = 24, max_train_size=None):
    arr = np.array(all_months)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        max_train_size=max_train_size
    )
    splits = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(arr)):
        train_months = arr[tr_idx].tolist()
        val_months = arr[va_idx].tolist()
        train_end = train_months[-1]
        splits.append((fold, train_end, train_months, val_months))
    return splits


def save_cv_results_csv(rows, csv_path):
    import csv
    if len(rows) == 0:
        return

    # Collect all keys that appear in rows so extra fields do not break CSV writing.
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_out = {k: r.get(k, None) for k in fieldnames}
            writer.writerow(row_out)


def append_row_csv(row, csv_path, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()


def save_summary_mean_sd_csv(rows, csv_path):
    if len(rows) == 0:
        return
    metric_keys = [
        "r2", "rmse", "mae", "bias", "agg_r2", "agg_rmse", "agg_mae", "agg_bias",
        "grad_mae", "locvar_ratio", "locvar_error", "pred_min", "pred_max",
        "frac_outside_3", "frac_outside_3_5", "frac_saturated_after_softbound",
    ]
    out_rows = []
    for key in metric_keys:
        vals = []
        for row in rows:
            try:
                v = float(row.get(key, np.nan))
            except Exception:
                v = np.nan
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out_rows.append({
                "metric": key,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=0)),
                "n": len(vals),
                "mean_sd": f"{float(np.mean(vals)):.6g} +/- {float(np.std(vals, ddof=0)):.6g}",
            })
        else:
            out_rows.append({"metric": key, "mean": "", "sd": "", "n": 0, "mean_sd": ""})
    save_cv_results_csv(out_rows, csv_path)


def save_cv_summary_json(rows, json_path):
    if len(rows) == 0:
        return

    keys_num = [
        "best_epoch", "best_val_loss", "full_val_loss", "full_val_rmse", "full_val_mae", "full_val_bias",
        "full_val_r2", "agg_r2", "agg_rmse", "agg_mae", "agg_bias", "grad_mae", "locvar_ratio",
        "locvar_error", "pred_min", "pred_max", "frac_outside_3", "frac_outside_3_5",
        "frac_saturated_after_softbound",
    ]
    summary = {}
    for k in keys_num:
        vals = []
        for r in rows:
            try:
                v = float(r.get(k, np.nan))
            except Exception:
                v = np.nan
            if np.isfinite(v):
                vals.append(v)
        if len(vals) == 0:
            summary[k] = {"mean": None, "std": None}
        else:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }

    valid_rows = []
    for r in rows:
        try:
            v = float(r.get("full_val_r2", np.nan))
        except Exception:
            v = np.nan
        if np.isfinite(v):
            valid_rows.append(r)
    if len(valid_rows) == 0:
        summary["best_fold_by_full_val_r2"] = None
    else:
        best_row = max(valid_rows, key=lambda x: x["full_val_r2"])
        summary["best_fold_by_full_val_r2"] = int(best_row["fold"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def audit_checkpoint_configs(search_roots, out_csv):
    fields = [
        "ckpt_path", "exp_tag", "model_name", "t_window", "w_pixel",
        "w_loss1", "w_loss2", "w_grad", "w_ms", "ms_scales", "convlstm_hidden_channels",
        "base_channels", "transformer_dim", "num_heads", "num_layers", "dropout_p",
        "seed", "loss_audit_label",
    ]
    rows = []
    for root in search_roots:
        if root is None or str(root).strip() == "" or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith((".pt", ".pth")):
                    continue
                ckpt_path = os.path.join(dirpath, fn)
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
                    w_loss1 = float(cfg.get("W_LOSS1", 0.0))
                    w_loss2 = float(cfg.get("W_LOSS2", 0.0))
                    rows.append({
                        "ckpt_path": ckpt_path,
                        "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
                        "model_name": "MS-TCFNet",
                        "t_window": cfg.get("T_WINDOW", ""),
                        "w_pixel": cfg.get("W_PIXEL", ""),
                        "w_loss1": w_loss1,
                        "w_loss2": w_loss2,
                        "w_grad": cfg.get("W_GRAD", ""),
                        "w_ms": cfg.get("W_MS", ""),
                        "ms_scales": cfg.get("MS_SCALES", ""),
                        "convlstm_hidden_channels": cfg.get("CONVLSTM_HIDDEN_CHANNELS", ""),
                        "base_channels": cfg.get("BASE_CHANNELS", ""),
                        "transformer_dim": cfg.get("TRANSFORMER_DIM", ""),
                        "num_heads": cfg.get("NUM_HEADS", ""),
                        "num_layers": cfg.get("NUM_LAYERS", ""),
                        "dropout_p": cfg.get("DROPOUT_P", ""),
                        "seed": cfg.get("SEED", ""),
                        "loss_audit_label": "L0L1L2",
                    })
                except Exception as e:
                    rows.append({"ckpt_path": ckpt_path, "loss_audit_label": f"load_failed: {e}"})

    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[CHECKPOINT AUDIT] {out_csv} (n={len(rows)})")


def configure_output_dirs(cfg):
    dirs = {
        "CHECKPOINT_DIR": os.path.join(cfg["OUT_DIR"], "checkpoints"),
        "METRICS_DIR": os.path.join(cfg["OUT_DIR"], "metrics"),
        "PREDICTION_DIR": os.path.join(cfg["OUT_DIR"], "predictions"),
        "SCATTER_DIR": os.path.join(cfg["OUT_DIR"], "scatter_samples"),
        "SPLIT_DIR": os.path.join(cfg["OUT_DIR"], "splits"),
        "STANDARDIZATION_DIR": os.path.join(cfg["OUT_DIR"], "standardization"),
        "LOG_DIR": os.path.join(cfg["OUT_DIR"], "logs"),
        "CACHE_DIR": os.path.join(cfg["OUT_DIR"], "sample_pool_cache"),
    }
    cfg.update(dirs)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return cfg


def save_split_json(path, cfg, stage_name, fold_id, train_months, val_months, train_end_tag="NA"):
    payload = {
        "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
        "stage": stage_name,
        "fold": int(fold_id),
        "train_end": train_end_tag,
        "train_months": list(train_months) if train_months is not None else None,
        "val_months": list(val_months) if val_months is not None else None,
        "n_train_months": len(train_months) if train_months is not None else None,
        "n_val_months": len(val_months) if val_months is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_standardization_params(mean, std, out_prefix, cfg, stage_name, fold_id):
    payload = {
        "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
        "stage": stage_name,
        "fold": int(fold_id),
        "mean": [float(x) for x in np.asarray(mean).reshape(-1)],
        "std": [float(x) for x in np.asarray(std).reshape(-1)],
    }
    json_path = out_prefix + ".json"
    npz_path = out_prefix + ".npz"
    os.makedirs(os.path.dirname(json_path), exist_ok=True) if os.path.dirname(json_path) else None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    np.savez_compressed(npz_path, mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))
    return json_path, npz_path


def tile_block_ids(tiles_xy, block_size_px):
    xy = np.asarray(tiles_xy, dtype=np.int64)
    x = xy[:, 0]
    y = xy[:, 1]
    bx = x // block_size_px
    by = y // block_size_px
    return (by * 100000 + bx).astype(np.int64)


def build_buffered_spatial_splits(sample_indices, sample_tile_xy, sample_groups, cfg):
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_tile_xy = np.asarray(sample_tile_xy, dtype=np.int64)
    sample_groups = np.asarray(sample_groups, dtype=np.int64)

    block_size = int(cfg["SBCV_BLOCK_SIZE_PX"])
    buffer_size = int(cfg.get("SBCV_BUFFER_SIZE_PX", 0))
    tile_size = int(cfg["TILE_SIZE"])

    gkf = GroupKFold(n_splits=cfg["SBCV_N_SPLITS"])
    splits = []

    tile_left = sample_tile_xy[:, 0]
    tile_top = sample_tile_xy[:, 1]
    tile_right = tile_left + tile_size
    tile_bottom = tile_top + tile_size

    for fold, (tr_idx_raw, va_idx) in enumerate(gkf.split(sample_indices, groups=sample_groups)):
        val_groups = np.unique(sample_groups[va_idx])
        train_candidate = np.ones(len(sample_indices), dtype=bool)
        train_candidate[va_idx] = False

        buffer_mask = np.zeros(len(sample_indices), dtype=bool)
        if buffer_size > 0:
            for group_id in val_groups.tolist():
                bx = int(group_id % 100000)
                by = int(group_id // 100000)
                left = bx * block_size - buffer_size
                right = (bx + 1) * block_size + buffer_size
                top = by * block_size - buffer_size
                bottom = (by + 1) * block_size + buffer_size

                intersects = (
                    (tile_right > left) &
                    (tile_left < right) &
                    (tile_bottom > top) &
                    (tile_top < bottom)
                )
                buffer_mask |= intersects

        tr_mask = train_candidate & (~buffer_mask)
        tr_idx = sample_indices[tr_mask]
        va_idx = sample_indices[va_idx]
        excluded_n = int((train_candidate & buffer_mask).sum())

        splits.append({
            "fold": int(fold),
            "train_idx": tr_idx.astype(np.int64),
            "val_idx": va_idx.astype(np.int64),
            "n_val_blocks": int(len(val_groups)),
            "n_buffer_excluded": excluded_n,
        })

    return splits


def build_local_indices_for_tiles(ds, chosen_tile_idx):
    n_files = len(ds.file_names)
    n_tiles = len(ds.tiles)
    chosen_tile_idx = np.asarray(chosen_tile_idx, dtype=np.int64)

    base = (np.arange(n_files, dtype=np.int64) * n_tiles)[:, None]
    idx = (base + chosen_tile_idx[None, :]).reshape(-1)
    return idx


# =========================================================
# 1) Dataset
# =========================================================
class MultiSourceDataFusionDataset(Dataset):
    def __init__(
            self,
            stack_dir,
            fine_label_dir,
            coarse_label_dir,
            yyyymm_list,
            tile_size,
            stride,
            t_window=6,
            nodata=-9999.0,
            use_coarse_loss=True,
            transform=None,
            min_valid_frac=0.0,
            sample_cache_path=None,
    ):
        super(MultiSourceDataFusionDataset, self).__init__()
        self.stack_dir = stack_dir
        self.fine_label_dir = fine_label_dir
        self.coarse_label_dir = coarse_label_dir
        self.yyyymm_list = list(yyyymm_list)
        self.tile_size = tile_size
        self.stride = stride
        self.t_window = t_window
        self.nodata = nodata
        self.use_coarse_loss = use_coarse_loss
        self.transform = transform
        self.min_valid_frac = float(min_valid_frac)
        self.sample_cache_path = sample_cache_path

        self._stack_cache = {}
        self._fine_cache = {}
        self._coarse_cache = {}

        self.file_names = self._get_time_steps()
        if len(self.file_names) == 0:
            raise RuntimeError("No valid time steps found for dataset.")

        self._check_fine_alignment()
        self.tiles = self._generate_tiles()

        self.samples = None
        self._load_or_build_sample_pool()

        self.coarse_patch_hw = None
        if self.use_coarse_loss:
            self._init_coarse_patch_shape()

    def _stack_path(self, year, month):
        return os.path.join(self.stack_dir, f"Stack_{year}_{month:02d}.tif")

    def _fine_path(self, year, month):
        return os.path.join(self.fine_label_dir, f"SPEI_{year}_{month:02d}.tif")

    def _coarse_path(self, year, month):
        return os.path.join(self.coarse_label_dir, f"SPEI_{year}_{month:02d}.tif")

    def _add_months(self, yyyymm, k):
        y = int(yyyymm[:4])
        m = int(yyyymm[4:6])
        total = y * 12 + (m - 1) + k
        yy = total // 12
        mm = total % 12 + 1
        return f"{yy:04d}{mm:02d}"

    def _open_cached(self, path, cache):
        ds = cache.get(path)
        if ds is None:
            ds = rasterio.open(path)
            cache[path] = ds
        return ds

    def close(self):
        for cache in [self._stack_cache, self._fine_cache, self._coarse_cache]:
            for _, ds in list(cache.items()):
                try:
                    ds.close()
                except Exception:
                    pass
            cache.clear()

    def reset_caches(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()

        # Rasterio handles cannot be pickled by DataLoader workers.
        for k in ["_stack_cache", "_fine_cache", "_coarse_cache"]:
            if k in state:
                state[k] = {}

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Reinitialize caches inside worker processes.
        self._stack_cache = {}
        self._fine_cache = {}
        self._coarse_cache = {}

    def _sample_pool_signature(self):
        sig = {
            "yyyymm_list": [rec["yyyymm"] for rec in self.file_names],
            "tile_size": int(self.tile_size),
            "stride": int(self.stride),
            "t_window": int(self.t_window),
            "nodata": float(self.nodata),
            "use_coarse_loss": int(self.use_coarse_loss),
            "min_valid_frac": float(self.min_valid_frac),
            "n_tiles": int(len(self.tiles)),
            "first_tile": tuple(self.tiles[0]) if len(self.tiles) > 0 else None,
            "last_tile": tuple(self.tiles[-1]) if len(self.tiles) > 0 else None,
        }
        sig_str = json.dumps(sig, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(sig_str.encode("utf-8")).hexdigest()

    def _save_sample_pool_cache(self):
        if self.sample_cache_path is None:
            return

        cache_dir = os.path.dirname(self.sample_cache_path)
        if cache_dir != "":
            os.makedirs(cache_dir, exist_ok=True)

        samples_arr = np.asarray(self.samples, dtype=np.int64)
        np.savez_compressed(
            self.sample_cache_path,
            signature=np.array(self._sample_pool_signature()),
            samples=samples_arr,
        )
        print(f"[CACHE] saved sample pool -> {self.sample_cache_path}")

    def _load_sample_pool_cache(self):
        if self.sample_cache_path is None:
            return False
        if not os.path.exists(self.sample_cache_path):
            return False

        try:
            data = np.load(self.sample_cache_path, allow_pickle=False)
            cache_sig = str(data["signature"].item())
            cur_sig = self._sample_pool_signature()

            if cache_sig != cur_sig:
                print(f"[CACHE] signature mismatch, rebuild sample pool: {self.sample_cache_path}")
                return False

            samples_arr = data["samples"]

            self.samples = [tuple(map(int, row)) for row in samples_arr.tolist()]

            print(f"[CACHE] loaded sample pool <- {self.sample_cache_path}")
            return True

        except Exception as e:
            print(f"[CACHE] failed to load cache, rebuild sample pool: {self.sample_cache_path} | {e}")
            return False

    def _load_or_build_sample_pool(self):
        loaded = self._load_sample_pool_cache()
        if loaded:
            return

        self.samples = self._generate_samples()
        self._save_sample_pool_cache()

    def _get_time_steps(self):
        time_steps = []
        for yyyymm in self.yyyymm_list:
            year = int(yyyymm[:4])
            month = int(yyyymm[4:6])

            fine_fp = self._fine_path(year, month)
            if not os.path.exists(fine_fp):
                continue

            if self.use_coarse_loss:
                coarse_fp = self._coarse_path(year, month)
                if not os.path.exists(coarse_fp):
                    continue
            else:
                coarse_fp = None

            ok = True
            seq_paths = []
            for lag in range(self.t_window - 1, -1, -1):
                mm = self._add_months(yyyymm, -lag)
                sy = int(mm[:4])
                sm = int(mm[4:6])
                sp = self._stack_path(sy, sm)
                if not os.path.exists(sp):
                    ok = False
                    break
                seq_paths.append(sp)

            if not ok:
                continue

            time_steps.append({
                "year": year,
                "month": month,
                "yyyymm": yyyymm,
                "stack_seq": seq_paths,
                "fine_label": fine_fp,
                "coarse_label": coarse_fp,
            })
        return time_steps

    def _check_fine_alignment(self):
        ref = self.file_names[0]
        with rasterio.open(ref["stack_seq"][-1]) as ds_stack, rasterio.open(ref["fine_label"]) as ds_fine:
            if str(ds_stack.crs) != str(ds_fine.crs):
                raise ValueError("Stack and 30m SPEI CRS mismatch.")

            if ds_stack.width != ds_fine.width or ds_stack.height != ds_fine.height:
                raise ValueError("Stack and 30m SPEI shape mismatch.")

            if not _transform_close(ds_stack.transform, ds_fine.transform, eps=1e-6):
                print("Stack transform:", tuple(ds_stack.transform))
                print("Fine  transform:", tuple(ds_fine.transform))
                raise ValueError("Stack and 30m SPEI transform mismatch.")

    def _generate_tiles(self):
        ref = self.file_names[0]
        with rasterio.open(ref["stack_seq"][-1]) as ds:
            H, W = ds.height, ds.width

        tiles = []
        xs = list(range(0, max(1, W - self.tile_size + 1), self.stride))
        ys = list(range(0, max(1, H - self.tile_size + 1), self.stride))

        if len(xs) == 0:
            xs = [0]
        if len(ys) == 0:
            ys = [0]

        if xs[-1] != W - self.tile_size:
            xs.append(max(0, W - self.tile_size))
        if ys[-1] != H - self.tile_size:
            ys.append(max(0, H - self.tile_size))

        for y0 in ys:
            for x0 in xs:
                tiles.append((x0, y0))
        return tiles

    def _generate_samples(self):
        samples = []

        for file_idx, rec in enumerate(self.file_names):
            yds = self._open_cached(rec["fine_label"], self._fine_cache)

            for tile_idx, (x0, y0) in enumerate(self.tiles):
                win = Window(x0, y0, self.tile_size, self.tile_size)

                if self.min_valid_frac > 0:
                    y = yds.read(1, window=win).astype(np.float32)
                    valid = (y != self.nodata) & (~np.isnan(y))
                    valid_frac = float(valid.mean())
                    if valid_frac < self.min_valid_frac:
                        continue

                samples.append((file_idx, tile_idx))

        return samples

    def _init_coarse_patch_shape(self):
        for rec in self.file_names:
            if rec["coarse_label"] is None:
                continue
            with rasterio.open(rec["stack_seq"][-1]) as fds, rasterio.open(rec["coarse_label"]) as cds:
                fine_w = abs(fds.transform.a) * self.tile_size
                fine_h = abs(fds.transform.e) * self.tile_size
                coarse_w = abs(cds.transform.a)
                coarse_h = abs(cds.transform.e)
                cw = max(1, int(round(fine_w / coarse_w)))
                ch = max(1, int(round(fine_h / coarse_h)))
                self.coarse_patch_hw = (ch, cw)
                return
        raise RuntimeError("Unable to initialize coarse patch shape.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, tile_idx = self.samples[idx]
        rec = self.file_names[file_idx]
        x0, y0 = self.tiles[tile_idx]
        win = Window(x0, y0, self.tile_size, self.tile_size)

        x_list = []
        for sp in rec["stack_seq"]:
            ds = self._open_cached(sp, self._stack_cache)
            arr = ds.read(window=win).astype(np.float32)
            x_list.append(arr)
        shapes = [tuple(x.shape) for x in x_list]
        if len(set(shapes)) != 1:
            detail = []
            for sp, arr in zip(rec["stack_seq"], x_list):
                try:
                    ds = self._open_cached(sp, self._stack_cache)
                    detail.append(f"{os.path.basename(sp)}: bands={ds.count}, shape={tuple(arr.shape)}")
                except Exception:
                    detail.append(f"{os.path.basename(sp)}: shape={tuple(arr.shape)}")
            raise RuntimeError(
                "Inconsistent Stack window shapes within one temporal sample. "
                f"yyyymm={rec['yyyymm']}, tile=(x0={x0}, y0={y0}), "
                f"details={detail}"
            )
        X = np.stack(x_list, axis=0)  # [T,C,H,W]

        yds = self._open_cached(rec["fine_label"], self._fine_cache)
        y_fine = yds.read(1, window=win).astype(np.float32)
        fine_mask = (y_fine != self.nodata) & (~np.isnan(y_fine))

        if self.use_coarse_loss:
            cds = self._open_cached(rec["coarse_label"], self._coarse_cache)
            left, bottom, right, top = rasterio.windows.bounds(win, yds.transform)
            cwin = window_from_bounds(left, bottom, right, top, cds.transform)
            ch, cw = self.coarse_patch_hw
            y_coarse = cds.read(
                1,
                window=cwin,
                out_shape=(ch, cw),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            coarse_mask = (y_coarse != self.nodata) & (~np.isnan(y_coarse))
        else:
            y_coarse = np.zeros((1, 1), dtype=np.float32)
            coarse_mask = np.zeros((1, 1), dtype=bool)

        if self.transform is not None:
            X, y_fine, fine_mask = self.transform(X, y_fine, fine_mask)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y_fine),
            torch.from_numpy(fine_mask),
            torch.from_numpy(y_coarse),
            torch.from_numpy(coarse_mask),
            rec["yyyymm"],
            x0,
            y0,
        )


def _sample_file_idx_for_local_index(dataset, local_idx):
    if hasattr(dataset, "samples"):
        return int(dataset.samples[int(local_idx)][0])
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "samples"):
        global_idx = int(dataset.indices[int(local_idx)])
        return int(dataset.dataset.samples[global_idx][0])
    return None


def _build_file_groups_for_dataset(dataset):
    file_to_indices = {}
    for local_idx in range(len(dataset)):
        file_idx = _sample_file_idx_for_local_index(dataset, local_idx)
        if file_idx is None:
            return None
        file_to_indices.setdefault(file_idx, []).append(local_idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in file_to_indices.items() if len(v) > 0}


class GroupedRandomSampler(Sampler):
    def __init__(self, dataset, num_samples, seed=42, replacement=True, batch_size=1, group_by_file=False):
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = seed
        self.replacement = replacement
        self._epoch = 0
        self.batch_size = max(1, int(batch_size))
        self.group_by_file = bool(group_by_file)
        self.file_to_indices = _build_file_groups_for_dataset(dataset) if self.group_by_file else None

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        if self.file_to_indices:
            file_ids = np.asarray(list(self.file_to_indices.keys()), dtype=np.int64)
            indices = []
            while len(indices) < self.num_samples:
                file_id = int(rng.choice(file_ids))
                pool = self.file_to_indices[file_id]
                take_n = min(self.batch_size, self.num_samples - len(indices))
                replace = self.replacement or take_n > len(pool)
                chosen = rng.choice(pool, size=take_n, replace=replace)
                indices.extend(chosen.tolist())
        else:
            indices = rng.choice(len(self.dataset), size=self.num_samples, replace=True).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


# =========================================================
# 2) Standardization
# =========================================================
class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, nodata: float, eps: float = 1e-6):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.nodata = nodata
        self.eps = eps

    def __call__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        valid_x = (X != self.nodata) & (~np.isnan(X))
        X_std = (X - self.mean[None, :, None, None]) / (self.std[None, :, None, None] + self.eps)
        X_out = np.zeros_like(X, dtype=np.float32)
        X_out[valid_x] = X_std[valid_x]
        return X_out, y, mask


def estimate_mean_std(dataset: Dataset, nodata: float, n_samples: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    C = None
    s = None
    s2 = None
    n = 0

    n_draw = min(n_samples, len(dataset))
    for _ in range(n_draw):
        j = int(rng.integers(0, len(dataset)))
        X, _, _, _, _, _, _, _ = dataset[j]
        Xn = X.numpy()
        valid_x = (Xn != nodata) & (~np.isnan(Xn))

        if C is None:
            C = Xn.shape[1]
            s = np.zeros((C,), dtype=np.float64)
            s2 = np.zeros((C,), dtype=np.float64)

        for c in range(C):
            vc = valid_x[:, c, :, :]
            if vc.any():
                vals = Xn[:, c, :, :][vc].astype(np.float64)
                s[c] += vals.mean()
                s2[c] += (vals * vals).mean()
        n += 1

    mean = s / max(n, 1)
    var = s2 / max(n, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# =========================================================
# 3) Models
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        xb = self.b(self.p3(x3))
        y3 = self.c3(torch.cat([self.u3(xb), x3], dim=1))
        y2 = self.c2(torch.cat([self.u2(y3), x2], dim=1))
        y1 = self.c1(torch.cat([self.u1(y2), x1], dim=1))
        return self.out(y1)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


def init_state(B, hid, H, W, device):
    h = torch.zeros(B, hid, H, W, device=device)
    c = torch.zeros(B, hid, H, W, device=device)
    return h, c


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pos_enc = TemporalPositionalEncoding(d_model=d_model, max_len=24, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        z_seq = self.encoder(x)
        z_last = z_seq[:, -1, :]
        return z_seq, z_last


class MSTCFNetBackbone(nn.Module):
    def __init__(self, in_ch, hid=64, base=32, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hid, d_model)
        self.ttr = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.film = nn.Linear(d_model, hid * 2)
        self.dec = UNet(hid, 1, base)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h, c = init_state(B, self.cell.hid_ch, H, W, x_seq.device)
        vecs = []
        for t in range(T):
            h, c = self.cell(x_seq[:, t], (h, c))
            v = self.pool(h).flatten(1)
            vecs.append(self.proj(v))

        vecs = torch.stack(vecs, dim=1)
        _, z = self.ttr(vecs)
        gamma, beta = torch.chunk(self.film(z), 2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = h * (1.0 + gamma) + beta
        return self.dec(h)


class FusionModel(nn.Module):
    def __init__(
            self,
            input_channels,
            convlstm_hidden_channels,
            base_channels,
            transformer_dim,
            num_heads,
            num_layers,
            dropout_p=0.1,
    ):
        super(FusionModel, self).__init__()
        self.model = MSTCFNetBackbone(
            in_ch=input_channels,
            hid=convlstm_hidden_channels,
            base=base_channels,
            d_model=transformer_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout_p,
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"MS-TCFNet expects x with shape [B,T,C,H,W], got {tuple(x.shape)}")
        return self.model(x)


# =========================================================
# 4) Loss / Metrics
# =========================================================
def _safe_masked_mean(x, mask, eps=1e-6):
    mask = mask.bool()
    valid_n = mask.sum()
    if valid_n.item() == 0:
        # Return graph-connected zero instead of a plain scalar zero.
        return x.sum() * 0.0
    return x[mask].mean()


def masked_huber(pred, target, mask, delta=1.0):
    if pred.dim() == 4:
        pred = pred[:, 0]
    mask = mask.bool()
    diff = pred - target
    absd = diff.abs()
    quad = torch.clamp(absd, max=delta)
    lin = absd - quad
    loss = 0.5 * quad * quad + delta * lin
    return _safe_masked_mean(loss, mask)


def downsample_area(x, size_hw):
    return F.interpolate(x.unsqueeze(1), size=size_hw, mode="area").squeeze(1)


def masked_gradient_loss(pred, target, mask):
    if pred.dim() == 4:
        pred = pred[:, 0]
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    tgt_dx = target[:, :, 1:] - target[:, :, :-1]
    tgt_dy = target[:, 1:, :] - target[:, :-1, :]
    mask_dx = mask[:, :, 1:] & mask[:, :, :-1]
    mask_dy = mask[:, 1:, :] & mask[:, :-1, :]
    loss_dx = _safe_masked_mean((pred_dx - tgt_dx).abs(), mask_dx)
    loss_dy = _safe_masked_mean((pred_dy - tgt_dy).abs(), mask_dy)
    return 0.5 * (loss_dx + loss_dy)


def masked_multiscale_loss(pred, target, mask, scales=(2, 4)):
    if pred.dim() == 4:
        pred = pred[:, 0]

    total = pred.new_tensor(0.0)
    count = 0
    for s in scales:
        H = pred.shape[-2] // s
        W = pred.shape[-1] // s
        if H < 2 or W < 2:
            continue

        p = downsample_area(pred, (H, W))
        t = downsample_area(target, (H, W))
        m = downsample_area(mask.float(), (H, W)) > 0.99
        total = total + masked_huber(p, t, m, delta=1.0)
        count += 1

    if count == 0:
        return pred.new_tensor(0.0)
    return total / count


def coarse_consistency_loss(pred, coarse_target, coarse_mask):
    if pred.dim() == 4:
        pred = pred[:, 0]
    Hc, Wc = coarse_target.shape[-2], coarse_target.shape[-1]
    pred_c = downsample_area(pred, (Hc, Wc))
    return masked_huber(pred_c, coarse_target, coarse_mask, delta=1.0)


def soft_bound(x, bound=3.5):
    return float(bound) * torch.tanh(x / float(bound))


def apply_output_policy(raw_pred, cfg, for_export=False):
    if cfg.get("USE_SOFT_BOUND_TRAIN", False) or (for_export and cfg.get("EXPORT_SOFT_BOUND", False)):
        return soft_bound(raw_pred, bound=cfg.get("Y_BOUND", 3.5))
    return raw_pred


@torch.no_grad()
def update_regression_acc(stats, pred, target, mask, prefix=""):
    if pred.dim() == 4:
        pred = pred[:, 0]
    mask = mask.bool()
    if mask.sum().item() == 0:
        return

    p = pred[mask].detach().float()
    t = target[mask].detach().float()
    diff = p - t
    stats[f"{prefix}sum_abs"] += diff.abs().sum().item()
    stats[f"{prefix}sum_sq"] += (diff * diff).sum().item()
    stats[f"{prefix}sum_diff"] += diff.sum().item()
    stats[f"{prefix}sum_t"] += t.sum().item()
    stats[f"{prefix}sum_t2"] += (t * t).sum().item()
    stats[f"{prefix}n"] += t.numel()


def finalize_regression_acc(stats, prefix=""):
    n = stats.get(f"{prefix}n", 0)
    if n <= 0:
        return np.nan, np.nan, np.nan, np.nan

    rmse = float(np.sqrt(stats[f"{prefix}sum_sq"] / n))
    mae = float(stats[f"{prefix}sum_abs"] / n)
    bias = float(stats[f"{prefix}sum_diff"] / n)
    ss_tot = float(stats[f"{prefix}sum_t2"] - (stats[f"{prefix}sum_t"] * stats[f"{prefix}sum_t"]) / n)
    r2 = float(1.0 - stats[f"{prefix}sum_sq"] / ss_tot) if ss_tot > 1e-12 else np.nan
    return rmse, mae, bias, r2


@torch.no_grad()
def update_structure_acc(stats, pred, target, mask, local_window=7):
    if pred.dim() == 4:
        pred = pred[:, 0]
    mask = mask.bool()

    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    tgt_dx = target[:, :, 1:] - target[:, :, :-1]
    tgt_dy = target[:, 1:, :] - target[:, :-1, :]
    mask_dx = mask[:, :, 1:] & mask[:, :, :-1]
    mask_dy = mask[:, 1:, :] & mask[:, :-1, :]

    if mask_dx.sum().item() > 0:
        stats["grad_abs_sum"] += (pred_dx - tgt_dx).abs()[mask_dx].sum().item()
        stats["grad_n"] += mask_dx.sum().item()
    if mask_dy.sum().item() > 0:
        stats["grad_abs_sum"] += (pred_dy - tgt_dy).abs()[mask_dy].sum().item()
        stats["grad_n"] += mask_dy.sum().item()

    k = int(local_window)
    if k < 3 or pred.shape[-2] < k or pred.shape[-1] < k:
        return
    if k % 2 == 0:
        k += 1
    pad = k // 2

    m = mask.float().unsqueeze(1)
    p = pred.float().unsqueeze(1)
    t = target.float().unsqueeze(1)
    valid_win = F.avg_pool2d(m, kernel_size=k, stride=1, padding=pad) > 0.999
    if valid_win.sum().item() == 0:
        return

    p_mean = F.avg_pool2d(p * m, kernel_size=k, stride=1, padding=pad)
    t_mean = F.avg_pool2d(t * m, kernel_size=k, stride=1, padding=pad)
    p_var = F.avg_pool2d((p * p) * m, kernel_size=k, stride=1, padding=pad) - p_mean * p_mean
    t_var = F.avg_pool2d((t * t) * m, kernel_size=k, stride=1, padding=pad) - t_mean * t_mean
    p_var = torch.clamp(p_var, min=0.0)
    t_var = torch.clamp(t_var, min=0.0)

    stats["locvar_pred_sum"] += p_var[valid_win].sum().item()
    stats["locvar_target_sum"] += t_var[valid_win].sum().item()
    stats["locvar_n"] += valid_win.sum().item()


def finalize_structure_acc(stats):
    grad_mae = float(stats["grad_abs_sum"] / stats["grad_n"]) if stats["grad_n"] > 0 else np.nan
    if stats["locvar_n"] > 0 and stats["locvar_target_sum"] > 1e-12:
        pred_mean = stats["locvar_pred_sum"] / stats["locvar_n"]
        target_mean = stats["locvar_target_sum"] / stats["locvar_n"]
        locvar_ratio = float(pred_mean / (target_mean + 1e-12))
        locvar_error = float(abs(locvar_ratio - 1.0))
    else:
        locvar_ratio = np.nan
        locvar_error = np.nan
    return grad_mae, locvar_ratio, locvar_error


@torch.no_grad()
def update_prediction_diagnostics(stats, pred, mask, y_bound=3.5):
    if pred.dim() == 4:
        pred = pred[:, 0]
    mask = mask.bool()
    if mask.sum().item() == 0:
        return

    vals = pred[mask].detach().float()
    stats["pred_min"] = min(stats["pred_min"], float(vals.min().item()))
    stats["pred_max"] = max(stats["pred_max"], float(vals.max().item()))
    stats["pred_n"] += vals.numel()
    stats["outside_3_n"] += (vals.abs() > 3.0).sum().item()
    stats["outside_3_5_n"] += (vals.abs() > 3.5).sum().item()
    bounded = soft_bound(vals, bound=y_bound)
    stats["saturated_n"] += (bounded.abs() > (0.99 * float(y_bound))).sum().item()


def finalize_prediction_diagnostics(stats):
    n = stats["pred_n"]
    if n <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        float(stats["pred_min"]),
        float(stats["pred_max"]),
        float(stats["outside_3_n"] / n),
        float(stats["outside_3_5_n"] / n),
        float(stats["saturated_n"] / n),
    )


@torch.no_grad()
def masked_rmse_mae_r2(pred, target, mask):
    if pred.dim() == 4:
        pred = pred[:, 0]
    mask = mask.bool()
    if mask.sum().item() == 0:
        return np.nan, np.nan, np.nan

    p = pred[mask]
    t = target[mask]
    diff = p - t
    rmse = torch.sqrt((diff * diff).mean()).item()
    mae = diff.abs().mean().item()
    t_mean = t.mean()
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t_mean) ** 2).sum()
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    return rmse, mae, r2


def compute_total_loss(pred, y_fine, fine_mask, y_coarse, coarse_mask, cfg):
    l0 = masked_huber(pred, y_fine, fine_mask, delta=cfg["HUBER_DELTA"])
    grad = masked_gradient_loss(pred, y_fine, fine_mask)
    ms = masked_multiscale_loss(pred, y_fine, fine_mask, scales=cfg["MS_SCALES"])
    l1 = cfg["W_GRAD"] * grad + cfg["W_MS"] * ms
    l2 = coarse_consistency_loss(pred, y_coarse, coarse_mask)

    total = cfg["W_PIXEL"] * l0 + cfg["W_LOSS1"] * l1 + cfg["W_LOSS2"] * l2
    return total, {
        "l0": float(l0.item()),
        "l1": float(l1.item()),
        "l2": float(l2.item()),
    }


# =========================================================
# 5) Trainer + Evaluator + EarlyStopping
# =========================================================
class ModelTrainer:
    def __init__(self, model, optimizer, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.use_amp = bool(cfg.get("USE_AMP", False)) and (
                str(device).startswith("cuda") or getattr(device, "type", "") == "cuda"
        )
        amp_dtype_name = str(cfg.get("AMP_DTYPE", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_one_epoch(self, loader, epoch=None):
        self.model.train()
        loss_list = []
        l0_list, l1_list, l2_list = [], [], []

        desc = f"Train E{epoch:03d}" if epoch is not None else "Train"
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)

        for X, y_fine, fine_mask, y_coarse, coarse_mask, *_ in pbar:
            X = X.to(self.device, non_blocking=True)
            y_fine = y_fine.to(self.device, non_blocking=True)
            fine_mask = fine_mask.to(self.device, non_blocking=True)
            y_coarse = y_coarse.to(self.device, non_blocking=True)
            coarse_mask = coarse_mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                raw_pred = self.model(X)
                pred = apply_output_policy(raw_pred, self.cfg)
                total, parts = compute_total_loss(
                    pred=pred,
                    y_fine=y_fine,
                    fine_mask=fine_mask,
                    y_coarse=y_coarse,
                    coarse_mask=coarse_mask,
                    cfg=self.cfg,
                )

            if (not torch.isfinite(total)) or (not total.requires_grad):
                pbar.set_postfix({
                    "loss": "invalid",
                    "l0": f"{parts['l0']:.4f}",
                    "l1": f"{parts['l1']:.4f}",
                    "l2": f"{parts['l2']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                continue

            self.scaler.scale(total).backward()

            if self.cfg["GRAD_CLIP"] is not None and self.cfg["GRAD_CLIP"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["GRAD_CLIP"])

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_list.append(float(total.item()))
            l0_list.append(float(parts["l0"]))
            l1_list.append(float(parts["l1"]))
            l2_list.append(float(parts["l2"]))

            pbar.set_postfix({
                "loss": f"{total.item():.4f}",
                "l0": f"{parts['l0']:.4f}",
                "l1": f"{parts['l1']:.4f}",
                "l2": f"{parts['l2']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        if len(loss_list) == 0:
            return {"loss": np.nan, "l0": np.nan, "l1": np.nan, "l2": np.nan}

        return {
            "loss": float(np.mean(loss_list)),
            "l0": float(np.mean(l0_list)),
            "l1": float(np.mean(l1_list)),
            "l2": float(np.mean(l2_list)),
        }


class ModelEvaluator:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.use_amp = bool(cfg.get("USE_AMP", False)) and (
                str(device).startswith("cuda") or getattr(device, "type", "") == "cuda"
        )
        amp_dtype_name = str(cfg.get("AMP_DTYPE", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16

    @torch.no_grad()
    def evaluate(self, loader, epoch=None, collect_rows=False, max_points=50000, n_bins=20, seed=42):
        self.model.eval()
        loss_list, l0_list, l1_list, l2_list = [], [], [], []

        # Global validation statistics.
        fine_stats = {"sum_abs": 0.0, "sum_sq": 0.0, "sum_diff": 0.0, "sum_t": 0.0, "sum_t2": 0.0, "n": 0}
        agg_stats = {"agg_sum_abs": 0.0, "agg_sum_sq": 0.0, "agg_sum_diff": 0.0,
                     "agg_sum_t": 0.0, "agg_sum_t2": 0.0, "agg_n": 0}
        struct_stats = {"grad_abs_sum": 0.0, "grad_n": 0, "locvar_pred_sum": 0.0,
                        "locvar_target_sum": 0.0, "locvar_n": 0}
        diag_stats = {"pred_min": float("inf"), "pred_max": float("-inf"), "pred_n": 0,
                      "outside_3_n": 0, "outside_3_5_n": 0, "saturated_n": 0}

        rng = np.random.default_rng(seed)
        rows = [] if collect_rows else None

        desc = f"Val E{epoch:03d}" if epoch is not None else "Val"
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)

        n_batches = max(len(loader), 1)
        per_batch_cap = max(100, max_points // n_batches) if collect_rows else 0

        def _to_int_scalar(v, idx=None):
            if torch.is_tensor(v):
                if v.ndim == 0:
                    return int(v.item())
                return int(v[idx].item())
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    return int(v.item())
                return int(v[idx])
            if isinstance(v, (list, tuple)):
                return int(v[idx])
            return int(v)

        def _pick_indices_by_quantile(target_arr, sample_cap, n_bins_local):
            if target_arr.size <= sample_cap:
                return np.arange(target_arr.size, dtype=np.int64)

            q = np.linspace(0, 1, n_bins_local + 1)
            edges = np.quantile(target_arr, q)
            edges = np.unique(edges)

            if edges.size < 3:
                return rng.choice(target_arr.size, size=sample_cap, replace=False).astype(np.int64)

            per_bin_cap = max(1, sample_cap // max(1, (len(edges) - 1)))
            chosen_idx = []

            for i in range(len(edges) - 1):
                if i == len(edges) - 2:
                    idx = np.where((target_arr >= edges[i]) & (target_arr <= edges[i + 1]))[0]
                else:
                    idx = np.where((target_arr >= edges[i]) & (target_arr < edges[i + 1]))[0]

                if idx.size == 0:
                    continue

                take_n = min(per_bin_cap, idx.size)
                chosen_idx.extend(rng.choice(idx, size=take_n, replace=False).tolist())

            if len(chosen_idx) == 0:
                return rng.choice(target_arr.size, size=sample_cap, replace=False).astype(np.int64)

            chosen_idx = np.asarray(chosen_idx, dtype=np.int64)

            if chosen_idx.size > sample_cap:
                chosen_idx = rng.choice(chosen_idx, size=sample_cap, replace=False)

            return chosen_idx

        for batch in pbar:
            X, y_fine, fine_mask, y_coarse, coarse_mask, *extras = batch

            X = X.to(self.device, non_blocking=True)
            y_fine = y_fine.to(self.device, non_blocking=True)
            fine_mask = fine_mask.to(self.device, non_blocking=True)
            y_coarse = y_coarse.to(self.device, non_blocking=True)
            coarse_mask = coarse_mask.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                raw_pred = self.model(X)
                pred = apply_output_policy(raw_pred, self.cfg)
                total, parts = compute_total_loss(
                    pred=pred,
                    y_fine=y_fine,
                    fine_mask=fine_mask,
                    y_coarse=y_coarse,
                    coarse_mask=coarse_mask,
                    cfg=self.cfg,
                )

            if not torch.isfinite(total):
                pbar.set_postfix({
                    "loss": "nan/inf",
                    "l0": f"{parts['l0']:.4f}",
                    "l1": f"{parts['l1']:.4f}",
                    "l2": f"{parts['l2']:.4f}",
                })
                continue

            loss_list.append(float(total.item()))
            l0_list.append(float(parts["l0"]))
            l1_list.append(float(parts["l1"]))
            l2_list.append(float(parts["l2"]))

            pred_ = pred[:, 0] if pred.dim() == 4 else pred
            mask_ = fine_mask.bool()

            update_regression_acc(fine_stats, pred_, y_fine, mask_)
            update_structure_acc(
                struct_stats,
                pred_,
                y_fine,
                mask_,
                local_window=self.cfg.get("LOCAL_VAR_WINDOW", 7),
            )
            update_prediction_diagnostics(
                diag_stats,
                pred_,
                mask_,
                y_bound=self.cfg.get("Y_BOUND", 3.5),
            )

            Hc, Wc = y_coarse.shape[-2], y_coarse.shape[-1]
            pred_c = downsample_area(pred_, (Hc, Wc))
            update_regression_acc(agg_stats, pred_c, y_coarse, coarse_mask, prefix="agg_")

            if collect_rows and len(extras) >= 3:
                yyyymm, x0, y0 = extras[-3], extras[-2], extras[-1]
                batch_size = pred_c.shape[0]
                per_sample_cap = max(10, per_batch_cap // max(batch_size, 1))

                for b in range(batch_size):
                    mb = coarse_mask[b].bool()
                    if mb.sum().item() == 0:
                        continue

                    p_grid = pred_c[b].detach().float().cpu().numpy()
                    t_grid = y_coarse[b].detach().float().cpu().numpy()
                    m_grid = mb.detach().cpu().numpy()
                    p_np = p_grid[m_grid]
                    t_np = t_grid[m_grid]

                    if t_np.size == 0:
                        continue

                    yy = yyyymm[b] if isinstance(yyyymm, (list, tuple)) else yyyymm
                    xx = _to_int_scalar(x0, b)
                    yy0 = _to_int_scalar(y0, b)
                    rr, cc = np.where(m_grid)

                    chosen_idx = _pick_indices_by_quantile(
                        target_arr=t_np,
                        sample_cap=per_sample_cap,
                        n_bins_local=n_bins,
                    )

                    for i in chosen_idx.tolist():
                        r = int(rr[i])
                        c = int(cc[i])
                        x_center = float(xx + (c + 0.5) * (self.cfg["TILE_SIZE"] / max(Wc, 1)))
                        y_center = float(yy0 + (r + 0.5) * (self.cfg["TILE_SIZE"] / max(Hc, 1)))
                        residual = float(p_np[i] - t_np[i])
                        rows.append({
                            "exp_tag": self.cfg.get("EXP_TAG", self.cfg.get("EXP_NAME", "")),
                            "stage": "",
                            "fold": "",
                            "yyyymm": yy,
                            "cell_id": f"{yy}_{xx}_{yy0}_{r}_{c}",
                            "x_center": x_center,
                            "y_center": y_center,
                            "target_1km_spei": float(t_np[i]),
                            "pred_agg_ehdi": float(p_np[i]),
                            "residual": residual,
                            "x0": xx,
                            "y0": yy0,
                            "pred": float(p_np[i]),
                            "target": float(t_np[i]),
                        })

            pbar.set_postfix({
                "loss": f"{total.item():.4f}",
                "l0": f"{parts['l0']:.4f}",
                "l1": f"{parts['l1']:.4f}",
                "l2": f"{parts['l2']:.4f}",
            })

        rmse, mae, bias, r2 = finalize_regression_acc(fine_stats)
        agg_rmse, agg_mae, agg_bias, agg_r2 = finalize_regression_acc(agg_stats, prefix="agg_")
        grad_mae, locvar_ratio, locvar_error = finalize_structure_acc(struct_stats)
        pred_min, pred_max, frac_outside_3, frac_outside_3_5, frac_saturated = finalize_prediction_diagnostics(
            diag_stats)

        if collect_rows and len(rows) > max_points:
            keep = rng.choice(len(rows), size=max_points, replace=False)
            rows = [rows[i] for i in keep.tolist()]

        return {
            "loss": float(np.mean(loss_list)) if len(loss_list) > 0 else np.nan,
            "l0": float(np.mean(l0_list)) if len(l0_list) > 0 else np.nan,
            "l1": float(np.mean(l1_list)) if len(l1_list) > 0 else np.nan,
            "l2": float(np.mean(l2_list)) if len(l2_list) > 0 else np.nan,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "r2": r2,
            "agg_rmse": agg_rmse,
            "agg_mae": agg_mae,
            "agg_bias": agg_bias,
            "agg_r2": agg_r2,
            "grad_mae": grad_mae,
            "locvar_ratio": locvar_ratio,
            "locvar_error": locvar_error,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "frac_outside_3": frac_outside_3,
            "frac_outside_3_5": frac_outside_3_5,
            "frac_saturated_after_softbound": frac_saturated,
            "rows": rows,
        }


SCATTER_FIELDNAMES = [
    "exp_tag",
    "stage",
    "fold",
    "yyyymm",
    "cell_id",
    "x_center",
    "y_center",
    "target_1km_spei",
    "pred_agg_ehdi",
    "residual",
    "x0",
    "y0",
    "pred",
    "target",
]


def save_points_cache_npz(rows, out_npz):
    if rows is None:
        return
    os.makedirs(os.path.dirname(out_npz), exist_ok=True) if os.path.dirname(out_npz) else None
    arrays = {}
    for field in SCATTER_FIELDNAMES:
        vals = [r.get(field, "") for r in rows]
        if field in {"exp_tag", "stage", "fold", "yyyymm", "cell_id"}:
            arrays[field] = np.asarray([str(v) for v in vals], dtype="<U128")
        elif field in {"x0", "y0"}:
            arrays[field] = np.asarray([int(v) if v != "" else -1 for v in vals], dtype=np.int32)
        else:
            arrays[field] = np.asarray([float(v) if v != "" else np.nan for v in vals], dtype=np.float32)
    np.savez_compressed(out_npz, **arrays)
    print(f"[CACHE] saved fullval points -> {out_npz}")


def write_points_csv_from_rows(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SCATTER_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SCATTER_FIELDNAMES})
    print(f"[SCATTER CSV] {out_csv} (n={len(rows)})")


def load_points_cache_npz(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    n = len(data["yyyymm"])

    rows = []
    for i in range(n):
        row = {}
        for field in SCATTER_FIELDNAMES:
            if field not in data.files:
                row[field] = ""
                continue
            v = data[field][i]
            if field in {"exp_tag", "stage", "fold", "yyyymm", "cell_id"}:
                row[field] = str(v)
            elif field in {"x0", "y0"}:
                row[field] = int(v)
            else:
                row[field] = float(v)
        rows.append(row)
    return rows


def _set_transform(ds, tfm):
    if isinstance(ds, Subset):
        ds.dataset.transform = tfm
    else:
        ds.transform = tfm


def _reset_caches(ds):
    base = ds.dataset if isinstance(ds, Subset) else ds
    if hasattr(base, "reset_caches"):
        base.reset_caches()


def _close_dataset(ds):
    base = ds.dataset if isinstance(ds, Subset) else ds
    if hasattr(base, "close"):
        base.close()


def dataloader_worker_kwargs(num_workers, cfg):
    if int(num_workers) <= 0:
        return {"num_workers": 0, "persistent_workers": False}
    return {
        "num_workers": int(num_workers),
        "persistent_workers": bool(cfg.get("PERSISTENT_WORKERS", True)),
        "multiprocessing_context": "spawn",
        "prefetch_factor": int(cfg.get("PREFETCH_FACTOR", 2)),
    }


def _make_model(cfg, input_channels):
    model = FusionModel(
        input_channels=input_channels,
        convlstm_hidden_channels=cfg["CONVLSTM_HIDDEN_CHANNELS"],
        base_channels=cfg["BASE_CHANNELS"],
        transformer_dim=cfg["TRANSFORMER_DIM"],
        num_heads=cfg["NUM_HEADS"],
        num_layers=cfg["NUM_LAYERS"],
        dropout_p=cfg["DROPOUT_P"],
    )
    return model.to(cfg["DEVICE"])


def _make_optimizer(cfg, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )


def _make_scheduler(cfg, optimizer):
    if cfg["USE_SCHEDULER"] and cfg["SCHEDULER_TYPE"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["PLATEAU_FACTOR"],
            patience=cfg["PLATEAU_PATIENCE"],
        )
    return None


HYPEROPT_SEARCH_CHOICES = {
    "CONVLSTM_HIDDEN_CHANNELS": [32, 64, 96, 128],
    "BASE_CHANNELS": [16, 32, 64],
    "TRANSFORMER_DIM": [128, 256, 512],
    "NUM_HEADS": [2, 4, 8],
    "NUM_LAYERS": [1, 2, 3, 4],
}


def build_mstcfnet_hyperopt_space():
    if hp is None:
        raise ImportError("hyperopt is required for MS-TCFNet Bayesian optimization.")

    return {
        "CONVLSTM_HIDDEN_CHANNELS": hp.choice(
            "CONVLSTM_HIDDEN_CHANNELS",
            HYPEROPT_SEARCH_CHOICES["CONVLSTM_HIDDEN_CHANNELS"],
        ),
        "BASE_CHANNELS": hp.choice(
            "BASE_CHANNELS",
            HYPEROPT_SEARCH_CHOICES["BASE_CHANNELS"],
        ),
        "TRANSFORMER_DIM": hp.choice(
            "TRANSFORMER_DIM",
            HYPEROPT_SEARCH_CHOICES["TRANSFORMER_DIM"],
        ),
        "NUM_HEADS": hp.choice(
            "NUM_HEADS",
            HYPEROPT_SEARCH_CHOICES["NUM_HEADS"],
        ),
        "NUM_LAYERS": hp.choice(
            "NUM_LAYERS",
            HYPEROPT_SEARCH_CHOICES["NUM_LAYERS"],
        ),
    }


def decode_mstcfnet_hyperopt_best(best):
    decoded = {}
    for key, choices in HYPEROPT_SEARCH_CHOICES.items():
        value = best.get(key)
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(choices):
            decoded[key] = choices[int(value)]
        else:
            decoded[key] = value
    decoded["DROPOUT_P"] = 0.1
    decoded["FFN_EXPANSION_RATIO"] = 4
    return decoded


def run_mstcfnet_hyperopt_bayesian_optimization(
        cfg,
        train_ds,
        val_ds,
        max_evals=30,
        max_epochs_per_trial=20,
        stage_name="HYPEROPT",
):
    """
    Archive utility for Hyperopt-based Bayesian optimization of MS-TCFNet.

    The final training pipeline does not call this function. It is retained to document and
    reproduce the architectural search used before fixing the final MS-TCFNet configuration.
    The search objective is full-validation loss on the supplied train/validation split.
    """
    if fmin is None or tpe is None or Trials is None:
        raise ImportError("Install hyperopt to run MS-TCFNet Bayesian optimization.")

    search_space = build_mstcfnet_hyperopt_space()
    trials = Trials()
    trial_counter = {"n": 0}

    def objective(params):
        trial_counter["n"] += 1
        trial_id = trial_counter["n"]
        trial_cfg = dict(cfg)
        trial_cfg.update({
            "CONVLSTM_HIDDEN_CHANNELS": int(params["CONVLSTM_HIDDEN_CHANNELS"]),
            "BASE_CHANNELS": int(params["BASE_CHANNELS"]),
            "TRANSFORMER_DIM": int(params["TRANSFORMER_DIM"]),
            "NUM_HEADS": int(params["NUM_HEADS"]),
            "NUM_LAYERS": int(params["NUM_LAYERS"]),
            "DROPOUT_P": 0.1,
            "EPOCHS": int(min(cfg["EPOCHS"], max_epochs_per_trial)),
            "EXP_TAG": f"{cfg['EXP_TAG']}_BO_trial{trial_id:03d}",
        })

        print(f"[HYPEROPT] trial={trial_id}, params={params}")
        try:
            row = _run_trainval_stage(
                stage_name=stage_name,
                fold_id=trial_id,
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=trial_cfg,
                train_end_tag="BAYESIAN_OPTIMIZATION",
                seed_offset=20000 + trial_id,
            )
            loss = float(row.get("full_val_loss", np.inf))
            status = STATUS_OK if np.isfinite(loss) else STATUS_FAIL
            return {"loss": loss, "status": status, "params": dict(params)}
        except Exception as exc:
            print(f"[HYPEROPT FAIL] trial={trial_id}: {exc}")
            return {"loss": float("inf"), "status": STATUS_FAIL, "params": dict(params)}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        trials=trials,
        rstate=np.random.default_rng(cfg.get("SEED", 42)),
    )
    best_params = decode_mstcfnet_hyperopt_best(best)
    print(f"[HYPEROPT BEST] raw={best}")
    print(f"[HYPEROPT BEST] decoded={best_params}")
    return best_params, trials


def loss_setting_from_cfg(cfg):
    return "L0L1L2"


def add_fullval_metrics_to_row(row, metrics, cfg, prediction_dir=None, scatter_csv=None):
    row.update({
        "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
        "model_name": "MS-TCFNet",
        "loss_setting": loss_setting_from_cfg(cfg),
        "r2": float(metrics.get("r2", np.nan)),
        "rmse": float(metrics.get("rmse", np.nan)),
        "mae": float(metrics.get("mae", np.nan)),
        "bias": float(metrics.get("bias", np.nan)),
        "agg_r2": float(metrics.get("agg_r2", np.nan)),
        "agg_rmse": float(metrics.get("agg_rmse", np.nan)),
        "agg_mae": float(metrics.get("agg_mae", np.nan)),
        "agg_bias": float(metrics.get("agg_bias", np.nan)),
        "grad_mae": float(metrics.get("grad_mae", np.nan)),
        "locvar_ratio": float(metrics.get("locvar_ratio", np.nan)),
        "locvar_error": float(metrics.get("locvar_error", np.nan)),
        "pred_min": float(metrics.get("pred_min", np.nan)),
        "pred_max": float(metrics.get("pred_max", np.nan)),
        "frac_outside_3": float(metrics.get("frac_outside_3", np.nan)),
        "frac_outside_3_5": float(metrics.get("frac_outside_3_5", np.nan)),
        "frac_saturated_after_softbound": float(metrics.get("frac_saturated_after_softbound", np.nan)),
    })
    if prediction_dir is not None:
        row["prediction_dir"] = prediction_dir
    if scatter_csv is not None:
        row["scatter_csv"] = scatter_csv
    return row


def annotate_scatter_rows(rows, cfg, stage_name, fold_id):
    if rows is None:
        return None
    exp_tag = cfg.get("EXP_TAG", cfg.get("EXP_NAME", ""))
    for r in rows:
        r["exp_tag"] = exp_tag
        r["stage"] = stage_name
        r["fold"] = int(fold_id)
    return rows


TRAINING_CURVE_FIELDS = [
    "exp_tag",
    "stage",
    "fold",
    "epoch",
    "train_loss",
    "train_l0",
    "train_l1",
    "train_l2",
    "val_loss",
    "val_l0",
    "val_l1",
    "val_l2",
    "val_rmse",
    "val_mae",
    "val_bias",
    "val_r2",
    "learning_rate",
    "best_so_far",
    "val_agg_rmse",
    "val_agg_r2",
    "val_grad_mae",
    "val_locvar_error",
    "pred_min",
    "pred_max",
    "frac_outside_3",
    "frac_outside_3_5",
    "last_ckpt",
    "best_ckpt",
]


def _run_trainval_stage(
        stage_name,
        fold_id,
        train_ds,
        val_ds,
        cfg,
        train_end_tag="NA",
        n_train_months=None,
        n_val_months=None,
        train_months_list=None,
        val_months_list=None,
        seed_offset=0,
):
    stage_tag = stage_name.lower()
    points_cache_npz = os.path.join(cfg["SCATTER_DIR"],
                                    f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_fullval_points_cache.npz")
    best_ckpt_path = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_best_fold{fold_id}.pt")
    last_ckpt_path = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_last_fold{fold_id}.pt")
    row_json_path = os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_fullval_metrics.json")
    points_csv = os.path.join(cfg["SCATTER_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_fullval_points.csv")
    split_json_path = os.path.join(cfg["SPLIT_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_split.json")
    std_prefix = os.path.join(cfg["STANDARDIZATION_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_standardization")
    training_curve_csv = os.path.join(cfg["LOG_DIR"], f"{cfg['EXP_TAG']}_{stage_tag}_fold{fold_id}_training_curve.csv")

    if train_months_list is not None or val_months_list is not None:
        save_split_json(split_json_path, cfg, stage_name, fold_id, train_months_list, val_months_list, train_end_tag)

    # =====================================================
    # 0) If metrics exist but CSV is missing, rebuild points CSV from cache.
    # =====================================================
    if os.path.exists(row_json_path):
        if (not os.path.exists(points_csv)) and os.path.exists(points_cache_npz):
            print(f"[RESUME] {stage_name} fold {fold_id}: rebuild points csv from cache.")
            rows = load_points_cache_npz(points_cache_npz)
            write_points_csv_from_rows(rows, points_csv)

            with open(row_json_path, "r", encoding="utf-8") as f:
                row = json.load(f)
            row["points_csv"] = points_csv
            row["scatter_csv"] = points_csv
            row["points_cache_npz"] = points_cache_npz

            with open(row_json_path, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)

            return row

    # =====================================================
    # A) If best checkpoint and validation artifacts exist, skip this fold.
    # =====================================================
    if os.path.exists(best_ckpt_path) and os.path.exists(row_json_path) and os.path.exists(points_csv):
        print(f"[RESUME] {stage_name} fold {fold_id}: best + fullval artifacts already exist, skip.")
        with open(row_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =====================================================
    # B) If the best checkpoint exists, skip training and finish validation if needed.
    # =====================================================
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=cfg["DEVICE"])

        # =====================================================
        # B2) Run full validation.
        # =====================================================
        print(f"[RESUME] {stage_name} fold {fold_id}: best checkpoint exists, skip training and run full validation.")

        if "mean" in ckpt and "std" in ckpt:
            save_standardization_params(ckpt["mean"], ckpt["std"], std_prefix, cfg, stage_name, fold_id)
        tfm = Standardizer(ckpt["mean"], ckpt["std"], cfg["NODATA"])
        _set_transform(val_ds, tfm)
        _reset_caches(val_ds)

        X0, _, _, _, _, _, _, _ = val_ds[0]
        _, C, _, _ = X0.shape

        model = _make_model(cfg, C)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_loader_full = DataLoader(
            val_ds,
            batch_size=cfg["VAL_BATCH_SIZE"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **dataloader_worker_kwargs(int(cfg.get("FULL_VAL_NUM_WORKERS", cfg.get("VAL_NUM_WORKERS", 2))), cfg),
        )

        evaluator = ModelEvaluator(model, cfg["DEVICE"], cfg)

        va_full = evaluator.evaluate(
            val_loader_full,
            epoch=None,
            collect_rows=True,
            max_points=cfg["SCATTER_MAX_POINTS"],
            n_bins=cfg["SCATTER_N_BINS"],
            seed=cfg["SEED"] + fold_id,
        )

        print(
            f"[{stage_name} Fold {fold_id} | FULL VAL] "
            f"loss={va_full['loss']:.4f} "
            f"l0={va_full['l0']:.4f} "
            f"l1={va_full['l1']:.4f} "
            f"l2={va_full['l2']:.4f} "
            f"r2={va_full['r2']:.4f} "
            f"agg_r2={va_full['agg_r2']:.4f} "
            f"grad_mae={va_full['grad_mae']:.4f} "
            f"locvar_err={va_full['locvar_error']:.4f} "
            f"locvar_ratio={va_full['locvar_ratio']:.4f}"
        )

        row = {
            "stage": stage_name,
            "fold": int(fold_id),
            "train_end": train_end_tag,
            "train_months": int(n_train_months) if n_train_months is not None else None,
            "val_months": int(n_val_months) if n_val_months is not None else None,
            "n_train_months": int(n_train_months) if n_train_months is not None else None,
            "n_val_months": int(n_val_months) if n_val_months is not None else None,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "full_val_loss": float(va_full["loss"]),
            "full_val_rmse": float(va_full["rmse"]),
            "full_val_mae": float(va_full["mae"]),
            "full_val_bias": float(va_full["bias"]),
            "full_val_r2": float(va_full["r2"]),
            "best_ckpt": best_ckpt_path,
            "last_ckpt": last_ckpt_path,
            "split_json": split_json_path,
            "standardization_json": ckpt.get("standardization_json", std_prefix + ".json"),
            "standardization_npz": ckpt.get("standardization_npz", std_prefix + ".npz"),
            "training_curve_csv": ckpt.get("training_curve_csv", training_curve_csv),
            "points_csv": None,
            "points_cache_npz": points_cache_npz,
            "result_source": "full_validation",
        }
        add_fullval_metrics_to_row(row, va_full, cfg, prediction_dir=None, scatter_csv=points_csv)

        with open(row_json_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        rows = annotate_scatter_rows(va_full["rows"], cfg, stage_name, fold_id)
        save_points_cache_npz(rows, points_cache_npz)
        write_points_csv_from_rows(rows, points_csv)

        row["points_csv"] = points_csv
        row["scatter_csv"] = points_csv
        with open(row_json_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        del val_loader_full, evaluator, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return row

    # =====================================================
    # C) Main training path.
    # =====================================================
    mean, std = estimate_mean_std(
        train_ds,
        cfg["NODATA"],
        n_samples=300,
        seed=cfg["SEED"] + seed_offset + 1000,
    )
    std_json_path, std_npz_path = save_standardization_params(mean, std, std_prefix, cfg, stage_name, fold_id)
    tfm = Standardizer(mean, std, cfg["NODATA"])
    _set_transform(train_ds, tfm)
    _set_transform(val_ds, tfm)
    _reset_caches(train_ds)
    _reset_caches(val_ds)

    X0, _, _, _, _, _, _, _ = train_ds[0]
    _, C, H, W = X0.shape
    print(f"[{stage_name} Fold {fold_id}] DATA C={C}, H={H}, W={W}, "
          f"n_train={len(train_ds)}, n_val={len(val_ds)}")

    # Shape probing opens rasterio handles; clear them before multiprocess loading.
    _reset_caches(train_ds)
    _reset_caches(val_ds)

    model = _make_model(cfg, C)
    optimizer = _make_optimizer(cfg, model)
    scheduler = _make_scheduler(cfg, optimizer)
    trainer = ModelTrainer(model, optimizer, cfg["DEVICE"], cfg)
    evaluator = ModelEvaluator(model, cfg["DEVICE"], cfg)
    early_stopper = EarlyStopping(
        patience=cfg["EARLY_STOP_PATIENCE"],
        delta=cfg["EARLY_STOP_DELTA"],
    )

    train_num_samples = min(cfg["TRAIN_SAMPLES_PER_EPOCH"], len(train_ds))
    train_sampler = GroupedRandomSampler(
        train_ds,
        num_samples=train_num_samples,
        seed=cfg["SEED"] + seed_offset,
        replacement=True,
        batch_size=cfg["TRAIN_BATCH_SIZE"],
        group_by_file=cfg.get("GROUP_BATCHES_BY_MONTH", True),
    )

    train_num_workers = cfg["NUM_WORKERS"]
    if stage_name == "SBCV":
        train_num_workers = cfg.get("SBCV_NUM_WORKERS", 0)

    if train_num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["TRAIN_BATCH_SIZE"],
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            **dataloader_worker_kwargs(train_num_workers, cfg),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["TRAIN_BATCH_SIZE"],
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            **dataloader_worker_kwargs(0, cfg),
        )

    # Fixed validation subset for epoch-level best monitoring; separate from full validation.
    rng = np.random.default_rng(cfg["SEED"] + 10000 + seed_offset + fold_id)
    n_val_monitor = min(cfg["VAL_SAMPLES_PER_EPOCH"], len(val_ds))
    if n_val_monitor <= 0:
        raise RuntimeError(f"{stage_name} fold {fold_id}: val_ds is empty.")

    monitor_indices = rng.choice(len(val_ds), size=n_val_monitor, replace=False).tolist()
    val_monitor_ds = Subset(val_ds, monitor_indices)

    val_num_workers = int(cfg.get("VAL_NUM_WORKERS", min(2, max(train_num_workers, 0))))
    val_loader = DataLoader(
        val_monitor_ds,
        batch_size=cfg["VAL_BATCH_SIZE"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        **dataloader_worker_kwargs(val_num_workers, cfg),
    )

    best_loss = np.inf
    best_epoch = -1

    for epoch in range(1, cfg["EPOCHS"] + 1):
        tr = trainer.train_one_epoch(train_loader, epoch=epoch)
        va = evaluator.evaluate(val_loader, epoch=epoch)

        if scheduler is not None and np.isfinite(va["loss"]):
            scheduler.step(va["loss"])

        print(
            f"[{stage_name} Fold {fold_id} | E{epoch:03d}] "
            f"train_loss={tr['loss']:.4f} "
            f"(l0={tr['l0']:.4f}, l1={tr['l1']:.4f}, l2={tr['l2']:.4f}) | "
            f"val_loss={va['loss']:.4f} "
            f"(l0={va['l0']:.4f}, l1={va['l1']:.4f}, l2={va['l2']:.4f}) | "
            f"r2={va['r2']:.4f} agg_r2={va['agg_r2']:.4f} | "
            f"grad_mae={va['grad_mae']:.4f} "
            f"locvar_err={va['locvar_error']:.4f} "
            f"locvar_ratio={va['locvar_ratio']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": trainer.scaler.state_dict() if hasattr(trainer, "scaler") else None,
            "config": cfg,
            "mean": mean,
            "std": std,
            "stage": stage_name,
            "fold": int(fold_id),
            "train_end": train_end_tag,
            "best_val_loss": float(best_loss) if np.isfinite(best_loss) else float("nan"),
            "standardization_json": std_json_path,
            "standardization_npz": std_npz_path,
            "split_json": split_json_path,
            "training_curve_csv": training_curve_csv,
        }
        torch.save(ckpt, last_ckpt_path)

        improved = np.isfinite(va["loss"]) and (va["loss"] < best_loss)
        if improved:
            best_loss = float(va["loss"])
            best_epoch = int(epoch)

            ckpt["best_val_loss"] = float(best_loss)
            ckpt["best_monitor_metrics"] = {
                "loss": float(va["loss"]),
                "l0": float(va["l0"]),
                "l1": float(va["l1"]),
                "l2": float(va["l2"]),
                "rmse": float(va["rmse"]),
                "mae": float(va["mae"]),
                "bias": float(va["bias"]),
                "r2": float(va["r2"]),
                "agg_rmse": float(va["agg_rmse"]),
                "agg_mae": float(va["agg_mae"]),
                "agg_bias": float(va["agg_bias"]),
                "agg_r2": float(va["agg_r2"]),
                "grad_mae": float(va["grad_mae"]),
                "locvar_ratio": float(va["locvar_ratio"]),
                "locvar_error": float(va["locvar_error"]),
                "pred_min": float(va["pred_min"]),
                "pred_max": float(va["pred_max"]),
                "frac_outside_3": float(va["frac_outside_3"]),
                "frac_outside_3_5": float(va["frac_outside_3_5"]),
                "frac_saturated_after_softbound": float(va["frac_saturated_after_softbound"]),
            }

            torch.save(ckpt, best_ckpt_path)

        history_row = {
            "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
            "stage": stage_name,
            "fold": int(fold_id),
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "train_l0": float(tr["l0"]),
            "train_l1": float(tr["l1"]),
            "train_l2": float(tr["l2"]),
            "val_loss": float(va["loss"]),
            "val_l0": float(va["l0"]),
            "val_l1": float(va["l1"]),
            "val_l2": float(va["l2"]),
            "val_rmse": float(va["rmse"]),
            "val_mae": float(va["mae"]),
            "val_bias": float(va["bias"]),
            "val_r2": float(va["r2"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_so_far": float(best_loss) if np.isfinite(best_loss) else "",
            "val_agg_rmse": float(va["agg_rmse"]),
            "val_agg_r2": float(va["agg_r2"]),
            "val_grad_mae": float(va["grad_mae"]),
            "val_locvar_error": float(va["locvar_error"]),
            "pred_min": float(va["pred_min"]),
            "pred_max": float(va["pred_max"]),
            "frac_outside_3": float(va["frac_outside_3"]),
            "frac_outside_3_5": float(va["frac_outside_3_5"]),
            "last_ckpt": last_ckpt_path,
            "best_ckpt": best_ckpt_path if os.path.exists(best_ckpt_path) else "",
        }
        append_row_csv(history_row, training_curve_csv, TRAINING_CURVE_FIELDS)

        early_stopper(va["loss"])
        print(
            f"[EARLY STOP] best={early_stopper.best_value:.6f} "
            f"bad_epochs={early_stopper.num_bad_epochs}/{early_stopper.patience}"
        )
        if early_stopper.stop:
            print(f"[EARLY STOP TRIGGERED] {stage_name} fold {fold_id}, epoch={epoch}")
            break

    del train_loader, val_loader, val_monitor_ds, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not os.path.exists(best_ckpt_path):
        raise RuntimeError(f"{stage_name} fold {fold_id}: training finished but best checkpoint not found.")

    print(f"[LOAD] {best_ckpt_path}")
    best = torch.load(best_ckpt_path, map_location=cfg["DEVICE"])

    model.load_state_dict(best["model_state"])
    model.eval()

    val_loader_full = DataLoader(
        val_ds,
        batch_size=cfg["VAL_BATCH_SIZE"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        **dataloader_worker_kwargs(int(cfg.get("FULL_VAL_NUM_WORKERS", cfg.get("VAL_NUM_WORKERS", 2))), cfg),
    )

    va_full = evaluator.evaluate(
        val_loader_full,
        epoch=None,
        collect_rows=True,
        max_points=cfg["SCATTER_MAX_POINTS"],
        n_bins=cfg["SCATTER_N_BINS"],
        seed=cfg["SEED"] + fold_id,
    )

    print(
        f"[{stage_name} Fold {fold_id} | FULL VAL] "
        f"loss={va_full['loss']:.4f} "
        f"l0={va_full['l0']:.4f} "
        f"l1={va_full['l1']:.4f} "
        f"l2={va_full['l2']:.4f} "
        f"r2={va_full['r2']:.4f} "
        f"agg_r2={va_full['agg_r2']:.4f} "
        f"grad_mae={va_full['grad_mae']:.4f} "
        f"locvar_err={va_full['locvar_error']:.4f} "
        f"locvar_ratio={va_full['locvar_ratio']:.4f}"
    )

    row = {
        "stage": stage_name,
        "fold": int(fold_id),
        "train_end": train_end_tag,
        "train_months": int(n_train_months) if n_train_months is not None else None,
        "val_months": int(n_val_months) if n_val_months is not None else None,
        "n_train_months": int(n_train_months) if n_train_months is not None else None,
        "n_val_months": int(n_val_months) if n_val_months is not None else None,
        "best_epoch": int(best_epoch if best_epoch >= 0 else best.get("epoch", -1)),
        "best_val_loss": float(best_loss if np.isfinite(best_loss) else best.get("best_val_loss", float("nan"))),
        "full_val_loss": float(va_full["loss"]),
        "full_val_rmse": float(va_full["rmse"]),
        "full_val_mae": float(va_full["mae"]),
        "full_val_bias": float(va_full["bias"]),
        "full_val_r2": float(va_full["r2"]),
        "best_ckpt": best_ckpt_path,
        "last_ckpt": last_ckpt_path,
        "split_json": split_json_path,
        "standardization_json": std_json_path,
        "standardization_npz": std_npz_path,
        "training_curve_csv": training_curve_csv,
        "points_csv": None,
        "points_cache_npz": points_cache_npz,
    }
    add_fullval_metrics_to_row(row, va_full, cfg, prediction_dir=None, scatter_csv=points_csv)

    # Save metrics before point export, then rewrite with CSV paths.
    with open(row_json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    rows = annotate_scatter_rows(va_full["rows"], cfg, stage_name, fold_id)
    save_points_cache_npz(rows, points_cache_npz)
    write_points_csv_from_rows(rows, points_csv)

    row["points_csv"] = points_csv
    row["scatter_csv"] = points_csv
    with open(row_json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    del val_loader_full, evaluator, model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_value = None
        self.num_bad_epochs = 0
        self.stop = False

    def __call__(self, value):
        if value is None or not np.isfinite(value):
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True
            return False

        if self.best_value is None:
            self.best_value = value
            self.num_bad_epochs = 0
            return True

        if value < self.best_value - self.delta:
            self.best_value = value
            self.num_bad_epochs = 0
            return True
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True
            return False


# =========================================================
# 6) Export
# =========================================================
def create_feather_mask_hann(tile_size, stride):
    overlap = tile_size - stride
    half_ov = overlap // 2
    w_1d = np.ones(tile_size, dtype=np.float32)

    if overlap > 0:
        hann_1d = np.hanning(overlap).astype(np.float32)
        w_1d[:half_ov] = hann_1d[:half_ov]
        w_1d[tile_size - half_ov:] = hann_1d[half_ov:]

    w_2d = np.outer(w_1d, w_1d).astype(np.float32)
    w_2d = np.maximum(w_2d, 1e-6)
    return w_2d


def build_ehdi_export_path(output_folder, cfg, stage_name, fold_id, yyyymm):
    exp_tag = cfg.get("EXP_TAG", cfg.get("EXP_NAME", "EXP"))
    variant = "softbound" if cfg.get("USE_SOFT_BOUND_TRAIN", False) or cfg.get("EXPORT_SOFT_BOUND", False) else "raw"
    return os.path.join(
        output_folder,
        f"EHDI_{exp_tag}_{stage_name}_fold{fold_id}_{yyyymm}_{variant}.tif",
    )


def select_best_fold_from_metrics(cfg, stage_name="TSCV"):
    stage = str(stage_name).upper()
    metric_name = cfg.get("EXPORT_BEST_FOLD_METRIC", "full_val_r2")

    if stage == "TSCV":
        metrics_csv = os.path.join(
            cfg["METRICS_DIR"],
            f"{cfg['EXP_TAG']}_tscv_metrics_fold.csv",
        )
        ckpt_prefix = "tscv"
    elif stage == "SBCV":
        metrics_csv = os.path.join(
            cfg["METRICS_DIR"],
            f"{cfg['EXP_TAG']}_sbcv_metrics_fold.csv",
        )
        ckpt_prefix = "sbcv"
    else:
        raise ValueError(f"Unsupported best-fold stage: {stage_name}")

    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(
            f"Cannot auto-select best {stage} fold because metrics CSV is missing:\n"
            f"  {metrics_csv}\n"
            "Run TSCV evaluation first."
        )

    best_row = None
    best_value = -np.inf
    with open(metrics_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_raw = row.get("fold", "")
            if fold_raw in ("", None):
                continue
            value_raw = row.get(metric_name, row.get("full_val_r2", row.get("r2", "")))
            try:
                value = float(value_raw)
                fold = int(float(fold_raw))
            except Exception:
                continue
            if np.isfinite(value) and value > best_value:
                best_value = value
                best_row = dict(row)
                best_row["fold"] = fold

    if best_row is None:
        raise RuntimeError(
            f"No finite {metric_name} values found in metrics CSV:\n"
            f"  {metrics_csv}"
        )

    fold_id = int(best_row["fold"])
    best_ckpt = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_{ckpt_prefix}_best_fold{fold_id}.pt")
    last_ckpt = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_{ckpt_prefix}_last_fold{fold_id}.pt")
    print(
        f"[EXPORT SELECT] {stage} best fold by {metric_name}: "
        f"fold={fold_id}, {metric_name}={best_value:.6f}"
    )
    return best_ckpt, last_ckpt, stage, fold_id

@torch.no_grad()
def export_fused_images(
        model,
        dataset,
        output_folder,
        yyyymm,
        device,
        nodata_value,
        dtype="float32",
        cfg=None,
        stage_name="FINAL",
        fold_id=0,
):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    cfg = cfg or {}

    tile_size = dataset.tile_size
    stride = dataset.stride
    feather_mask_full = create_feather_mask_hann(tile_size, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    ref_rec = dataset.file_names[0]
    ref_path = ref_rec["stack_seq"][-1]
    with rasterio.open(ref_path) as ref_ds:
        height, width = ref_ds.height, ref_ds.width
        meta = ref_ds.meta.copy()

    fused_image = np.zeros((height, width), dtype=np.float32)
    weight_matrix = np.zeros((height, width), dtype=np.float32)

    use_amp = (str(device).startswith("cuda") or getattr(device, "type", "") == "cuda")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
        for batch in dataloader:
            X, _, fine_mask, _, _, _, x0, y0 = batch
            X = X.to(device, non_blocking=True)
            fine_mask = fine_mask.to(device, non_blocking=True)

            raw_pred = model(X)
            pred = apply_output_policy(raw_pred, cfg, for_export=True)
            pred_np = pred.squeeze().detach().float().cpu().numpy().astype(np.float32)
            mask_np = fine_mask.squeeze().detach().float().cpu().numpy().astype(np.float32)

            x0 = int(x0.item()) if torch.is_tensor(x0) else int(x0)
            y0 = int(y0.item()) if torch.is_tensor(y0) else int(y0)

            local_h = min(tile_size, height - y0)
            local_w = min(tile_size, width - x0)

            pred_np = pred_np[:local_h, :local_w]
            mask_np = mask_np[:local_h, :local_w]
            feather = feather_mask_full[:local_h, :local_w]

            local_weight = feather * mask_np
            fused_image[y0:y0 + local_h, x0:x0 + local_w] += pred_np * local_weight
            weight_matrix[y0:y0 + local_h, x0:x0 + local_w] += local_weight

    valid = weight_matrix > 0
    fused_image[valid] = fused_image[valid] / weight_matrix[valid]
    fused_image[~valid] = nodata_value

    meta.update(
        count=1,
        dtype=dtype,
        nodata=nodata_value,
        compress="deflate",
        tiled=True,
    )

    out_path = build_ehdi_export_path(output_folder, cfg, stage_name, fold_id, yyyymm)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(fused_image.astype(dtype), 1)

    print(f"[EXPORT] {out_path}")
    return out_path


# =========================================================
# 7) Main
# =========================================================
if __name__ == "__main__":
    cfg = dict(
        STACK_DIR=r"/root/autodl-tmp/Stack",
        FINE_LABEL_DIR=r"/root/autodl-tmp/SPEI_30M",
        COARSE_LABEL_DIR=r"/root/autodl-tmp/SPEI_1KM",
        NODATA=-9999.0,
        SEED=42,
        ROOT_OUT_DIR=r"/root/autodl-fs/EHDI",

        EXP_NAME="MS-TCFNet",
        EXP_TAG="B2_L0L1L2",
        MODEL_NAME="MS-TCFNet",

        DEVICE="cuda" if torch.cuda.is_available() else "cpu",
        TILE_SIZE=256,
        STRIDE=128,
        T_WINDOW=6,
        CONVLSTM_HIDDEN_CHANNELS=64,
        BASE_CHANNELS=32,
        TRANSFORMER_DIM=256,
        NUM_HEADS=4,
        NUM_LAYERS=2,
        DROPOUT_P=0.1,

        EPOCHS=100,
        TRAIN_BATCH_SIZE=12,
        VAL_BATCH_SIZE=8,
        NUM_WORKERS=2,
        VAL_NUM_WORKERS=0,
        FULL_VAL_NUM_WORKERS=0,
        PERSISTENT_WORKERS=False,
        PREFETCH_FACTOR=1,
        GROUP_BATCHES_BY_MONTH=False,
        TRAIN_SAMPLES_PER_EPOCH=2000,
        VAL_SAMPLES_PER_EPOCH=800,

        LR=2e-4,
        WEIGHT_DECAY=1e-4,
        MIN_VALID_FRAC=0.30,
        GRAD_CLIP=1.0,
        USE_AMP=True,
        AMP_DTYPE="float16",
        USE_SCHEDULER=True,
        SCHEDULER_TYPE="plateau",
        PLATEAU_FACTOR=0.5,
        PLATEAU_PATIENCE=2,
        EARLY_STOP_PATIENCE=10,
        EARLY_STOP_DELTA=1e-4,

        HUBER_DELTA=1.0,
        W_PIXEL=1.0,
        W_LOSS1=1.0,
        W_LOSS2=1.0,
        W_GRAD=1.0,
        W_MS=1.0,
        MS_SCALES=(2, 4),
        LOCAL_VAR_WINDOW=7,
        USE_SOFT_BOUND_TRAIN=False,

        START="200001",
        END="202412",
        TSCV_N_SPLITS=5,
        TSCV_TEST_SIZE=24,
        TSCV_MAX_TRAIN_SIZE=180,
        SBCV_N_SPLITS=5,
        SBCV_STRIDE=256,
        SBCV_NUM_WORKERS=2,
        SBCV_STAGE_NAME="SBCV",
        SBCV_BLOCK_SIZE_PX=1024,
        SBCV_BUFFER_SIZE_PX=512,

        FINAL_EARLY_STOP_PATIENCE=10,
        FINAL_EARLY_STOP_DELTA=1e-4,
        SCATTER_MAX_POINTS=50000,
        SCATTER_N_BINS=20,
        EXPORT_SOFT_BOUND=True,
        Y_BOUND=3.0,
        EXPORT_BEST_FOLD_METRIC="full_val_r2",
    )

    cfg["OUT_DIR"] = os.path.join(cfg["ROOT_OUT_DIR"], cfg["EXP_TAG"])
    configure_output_dirs(cfg)

    with open(os.path.join(cfg["OUT_DIR"], "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    set_seed(cfg["SEED"])
    sample_cache_dir = cfg["CACHE_DIR"]

    def make_sample_cache_path(stage, fold_id, split_name):
        return os.path.join(sample_cache_dir, f"{stage.lower()}_fold{fold_id}_{split_name}_pool.npz")

    ensure_exists(cfg["STACK_DIR"], "STACK_DIR")
    ensure_exists(cfg["FINE_LABEL_DIR"], "FINE_LABEL_DIR")
    ensure_exists(cfg["COARSE_LABEL_DIR"], "COARSE_LABEL_DIR")

    all_months = list_months(cfg["START"], cfg["END"])
    available_months = []
    for yyyymm in all_months:
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])
        fine_fp = os.path.join(cfg["FINE_LABEL_DIR"], f"SPEI_{year}_{month:02d}.tif")
        coarse_fp = os.path.join(cfg["COARSE_LABEL_DIR"], f"SPEI_{year}_{month:02d}.tif")
        if not os.path.exists(fine_fp) or not os.path.exists(coarse_fp):
            continue

        ok = True
        for lag in range(cfg["T_WINDOW"] - 1, -1, -1):
            mm = add_months(yyyymm, -lag)
            sp = os.path.join(cfg["STACK_DIR"], f"Stack_{int(mm[:4])}_{int(mm[4:6]):02d}.tif")
            if not os.path.exists(sp):
                ok = False
                break
        if ok:
            available_months.append(yyyymm)

    if len(available_months) == 0:
        raise RuntimeError("No valid available months found.")

    splits = build_tscv_month_splits(
        available_months,
        n_splits=cfg["TSCV_N_SPLITS"],
        test_size=cfg["TSCV_TEST_SIZE"],
        max_train_size=cfg.get("TSCV_MAX_TRAIN_SIZE", None),
    )
    if len(splits) == 0:
        raise RuntimeError("No TSCV splits generated. Check available months / split config.")

    tscv_rows = []
    for fold, train_end, train_months, val_months in splits:
        print(f"[TSCV] fold={fold}, train_end={train_end}, train_months={len(train_months)}, val_months={len(val_months)}")
        train_ds = MultiSourceDataFusionDataset(
            stack_dir=cfg["STACK_DIR"],
            fine_label_dir=cfg["FINE_LABEL_DIR"],
            coarse_label_dir=cfg["COARSE_LABEL_DIR"],
            yyyymm_list=train_months,
            tile_size=cfg["TILE_SIZE"],
            stride=cfg["STRIDE"],
            t_window=cfg["T_WINDOW"],
            nodata=cfg["NODATA"],
            use_coarse_loss=True,
            transform=None,
            min_valid_frac=cfg["MIN_VALID_FRAC"],
            sample_cache_path=make_sample_cache_path("TSCV", fold + 1, "train"),
        )
        val_ds = MultiSourceDataFusionDataset(
            stack_dir=cfg["STACK_DIR"],
            fine_label_dir=cfg["FINE_LABEL_DIR"],
            coarse_label_dir=cfg["COARSE_LABEL_DIR"],
            yyyymm_list=val_months,
            tile_size=cfg["TILE_SIZE"],
            stride=cfg["STRIDE"],
            t_window=cfg["T_WINDOW"],
            nodata=cfg["NODATA"],
            use_coarse_loss=True,
            transform=None,
            min_valid_frac=cfg["MIN_VALID_FRAC"],
            sample_cache_path=make_sample_cache_path("TSCV", fold + 1, "val"),
        )
        row = _run_trainval_stage(
            stage_name="TSCV",
            fold_id=fold + 1,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            train_end_tag=train_end,
            n_train_months=len(train_months),
            n_val_months=len(val_months),
            train_months_list=train_months,
            val_months_list=val_months,
            seed_offset=fold,
        )
        tscv_rows.append(row)
        _close_dataset(train_ds)
        _close_dataset(val_ds)

    tscv_csv = os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_tscv_metrics_fold.csv")
    tscv_json = os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_tscv_summary.json")
    tscv_summary_csv = os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_tscv_summary_mean_sd.csv")
    save_cv_results_csv(tscv_rows, tscv_csv)
    save_cv_summary_json(tscv_rows, tscv_json)
    save_summary_mean_sd_csv(tscv_rows, tscv_summary_csv)
    print(f"[TSCV RESULTS] {tscv_csv}")
    print(f"[TSCV SUMMARY] {tscv_json}")

    print("[SBCV] start...")
    sbcv_rows = []
    all_ds = MultiSourceDataFusionDataset(
        stack_dir=cfg["STACK_DIR"],
        fine_label_dir=cfg["FINE_LABEL_DIR"],
        coarse_label_dir=cfg["COARSE_LABEL_DIR"],
        yyyymm_list=available_months,
        tile_size=cfg["TILE_SIZE"],
        stride=cfg["SBCV_STRIDE"],
        t_window=cfg["T_WINDOW"],
        nodata=cfg["NODATA"],
        use_coarse_loss=True,
        transform=None,
        min_valid_frac=cfg["MIN_VALID_FRAC"],
        sample_cache_path=os.path.join(sample_cache_dir, "sbcv_all_pool.npz"),
    )
    sample_tile_idx = np.array([tile_idx for _, tile_idx in all_ds.samples], dtype=np.int64)
    sample_tile_xy = np.asarray([all_ds.tiles[t] for t in sample_tile_idx], dtype=np.int64)
    sample_groups = tile_block_ids(sample_tile_xy, cfg["SBCV_BLOCK_SIZE_PX"])
    sample_indices = np.arange(len(all_ds), dtype=np.int64)
    sbcv_stage_name = cfg.get("SBCV_STAGE_NAME", "SBCV")

    if int(cfg.get("SBCV_BUFFER_SIZE_PX", 0)) > 0:
        sbcv_split_plan = build_buffered_spatial_splits(sample_indices, sample_tile_xy, sample_groups, cfg)
    else:
        gkf = GroupKFold(n_splits=cfg["SBCV_N_SPLITS"])
        sbcv_split_plan = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(sample_indices, groups=sample_groups)):
            sbcv_split_plan.append({
                "fold": int(fold),
                "train_idx": np.asarray(tr_idx, dtype=np.int64),
                "val_idx": np.asarray(va_idx, dtype=np.int64),
                "n_val_blocks": int(len(np.unique(sample_groups[va_idx]))),
                "n_buffer_excluded": 0,
            })

    for split_info in sbcv_split_plan:
        fold = int(split_info["fold"])
        tr_idx = split_info["train_idx"]
        va_idx = split_info["val_idx"]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise RuntimeError(f"{sbcv_stage_name} fold {fold}: empty train/val split after buffering.")
        row = _run_trainval_stage(
            stage_name=sbcv_stage_name,
            fold_id=fold + 1,
            train_ds=Subset(all_ds, tr_idx.tolist()),
            val_ds=Subset(all_ds, va_idx.tolist()),
            cfg=cfg,
            train_end_tag="SPACE_BLOCK_BUFFERED" if int(cfg.get("SBCV_BUFFER_SIZE_PX", 0)) > 0 else "SPACE_BLOCK",
            n_train_months=len(available_months),
            n_val_months=len(available_months),
            train_months_list=available_months,
            val_months_list=available_months,
            seed_offset=100 + fold,
        )
        sbcv_rows.append(row)
    _close_dataset(all_ds)

    sbcv_tag = sbcv_stage_name.lower()
    save_cv_results_csv(sbcv_rows, os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_{sbcv_tag}_metrics_fold.csv"))
    save_cv_summary_json(sbcv_rows, os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_{sbcv_tag}_summary.json"))
    save_summary_mean_sd_csv(sbcv_rows, os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_{sbcv_tag}_summary_mean_sd.csv"))

    all_eval_rows = tscv_rows + sbcv_rows
    save_cv_results_csv(all_eval_rows, os.path.join(cfg["METRICS_DIR"], "all_experiments_fold_metrics.csv"))
    save_summary_mean_sd_csv(all_eval_rows, os.path.join(cfg["METRICS_DIR"], "all_experiments_summary_mean_sd.csv"))

    print("[FINAL RETRAIN] start...")
    final_months = available_months
    final_best_ckpt = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_final_all_data_best.pt")
    final_last_ckpt = os.path.join(cfg["CHECKPOINT_DIR"], f"{cfg['EXP_TAG']}_final_all_data_last.pt")
    final_metrics_json = os.path.join(cfg["METRICS_DIR"], f"{cfg['EXP_TAG']}_final_retrain_metrics.json")
    final_split_json = os.path.join(cfg["SPLIT_DIR"], f"{cfg['EXP_TAG']}_final_split.json")
    final_std_prefix = os.path.join(cfg["STANDARDIZATION_DIR"], f"{cfg['EXP_TAG']}_final_standardization")
    final_training_curve_csv = os.path.join(cfg["LOG_DIR"], f"{cfg['EXP_TAG']}_final_training_curve.csv")
    export_folder = cfg["PREDICTION_DIR"]
    export_months = final_months
    save_split_json(final_split_json, cfg, "FINAL_RETRAIN", 0, final_months, final_months, "ALL_AVAILABLE")

    if os.path.exists(final_best_ckpt) and os.path.exists(final_metrics_json):
        print("[RESUME] FINAL RETRAIN: best + metrics already exist, skip training.")
        with open(final_metrics_json, "r", encoding="utf-8") as f:
            final_row = json.load(f)
    else:
        final_ds = MultiSourceDataFusionDataset(
            stack_dir=cfg["STACK_DIR"],
            fine_label_dir=cfg["FINE_LABEL_DIR"],
            coarse_label_dir=cfg["COARSE_LABEL_DIR"],
            yyyymm_list=final_months,
            tile_size=cfg["TILE_SIZE"],
            stride=cfg["STRIDE"],
            t_window=cfg["T_WINDOW"],
            nodata=cfg["NODATA"],
            use_coarse_loss=True,
            transform=None,
            min_valid_frac=cfg["MIN_VALID_FRAC"],
            sample_cache_path=os.path.join(sample_cache_dir, "final_retrain_all_pool.npz"),
        )
        mean, std = estimate_mean_std(final_ds, cfg["NODATA"], n_samples=300, seed=cfg["SEED"] + 999)
        final_std_json, final_std_npz = save_standardization_params(mean, std, final_std_prefix, cfg, "FINAL_RETRAIN", 0)
        final_ds.transform = Standardizer(mean, std, cfg["NODATA"])

        X0, _, _, _, _, _, _, _ = final_ds[0]
        T, C, H, W = X0.shape
        print(f"[FINAL RETRAIN] DATA T={T}, C={C}, H={H}, W={W}")
        final_ds.reset_caches()

        model = _make_model(cfg, C)
        optimizer = _make_optimizer(cfg, model)
        scheduler = _make_scheduler(cfg, optimizer)
        final_sampler = GroupedRandomSampler(
            final_ds,
            num_samples=min(cfg["TRAIN_SAMPLES_PER_EPOCH"], len(final_ds)),
            seed=cfg["SEED"] + 999,
            replacement=True,
            batch_size=cfg["TRAIN_BATCH_SIZE"],
            group_by_file=cfg.get("GROUP_BATCHES_BY_MONTH", True),
        )
        final_loader = DataLoader(
            final_ds,
            batch_size=cfg["TRAIN_BATCH_SIZE"],
            sampler=final_sampler,
            pin_memory=True,
            drop_last=True,
            **dataloader_worker_kwargs(cfg["NUM_WORKERS"], cfg),
        )

        final_best_loss = np.inf
        final_best_epoch = -1
        trainer = ModelTrainer(model, optimizer, cfg["DEVICE"], cfg)
        final_early_stopper = EarlyStopping(
            patience=cfg.get("FINAL_EARLY_STOP_PATIENCE", cfg["EARLY_STOP_PATIENCE"]),
            delta=cfg.get("FINAL_EARLY_STOP_DELTA", cfg["EARLY_STOP_DELTA"]),
        )

        for epoch in range(1, cfg["EPOCHS"] + 1):
            tr = trainer.train_one_epoch(final_loader, epoch=epoch)
            if scheduler is not None and np.isfinite(tr["loss"]):
                scheduler.step(tr["loss"])
            print(f"[FINAL RETRAIN | E{epoch:03d}] train_loss={tr['loss']:.4f} (l0={tr['l0']:.4f}, l1={tr['l1']:.4f}, l2={tr['l2']:.4f})")

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": trainer.scaler.state_dict() if hasattr(trainer, "scaler") else None,
                "config": cfg,
                "mean": mean,
                "std": std,
                "stage": "FINAL_RETRAIN",
                "best_train_loss": float(final_best_loss) if np.isfinite(final_best_loss) else float("nan"),
                "standardization_json": final_std_json,
                "standardization_npz": final_std_npz,
                "split_json": final_split_json,
                "training_curve_csv": final_training_curve_csv,
            }
            torch.save(ckpt, final_last_ckpt)

            improved = np.isfinite(tr["loss"]) and (tr["loss"] < final_best_loss)
            if improved:
                final_best_loss = float(tr["loss"])
                final_best_epoch = int(epoch)
                ckpt["best_train_loss"] = float(final_best_loss)
                torch.save(ckpt, final_best_ckpt)

            final_history_row = {
                "exp_tag": cfg.get("EXP_TAG", cfg.get("EXP_NAME", "")),
                "stage": "FINAL_RETRAIN",
                "fold": 0,
                "epoch": int(epoch),
                "train_loss": float(tr["loss"]),
                "train_l0": float(tr["l0"]),
                "train_l1": float(tr["l1"]),
                "train_l2": float(tr["l2"]),
                "val_loss": "",
                "val_l0": "",
                "val_l1": "",
                "val_l2": "",
                "val_rmse": "",
                "val_mae": "",
                "val_bias": "",
                "val_r2": "",
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_so_far": float(final_best_loss) if np.isfinite(final_best_loss) else "",
                "last_ckpt": final_last_ckpt,
                "best_ckpt": final_best_ckpt if os.path.exists(final_best_ckpt) else "",
            }
            append_row_csv(final_history_row, final_training_curve_csv, TRAINING_CURVE_FIELDS)

            final_early_stopper(tr["loss"])
            print(f"[FINAL EARLY STOP] best={final_early_stopper.best_value:.6f} bad_epochs={final_early_stopper.num_bad_epochs}/{final_early_stopper.patience}")
            if final_early_stopper.stop:
                print(f"[FINAL EARLY STOP TRIGGERED] epoch={epoch}")
                break

        print(f"[FINAL RETRAIN DONE] best_train_loss={final_best_loss:.6f}, best_epoch={final_best_epoch}")
        final_row = {
            "stage": "FINAL_RETRAIN",
            "best_epoch": int(final_best_epoch),
            "best_train_loss": float(final_best_loss),
            "best_ckpt": final_best_ckpt,
            "last_ckpt": final_last_ckpt,
            "split_json": final_split_json,
            "standardization_json": final_std_json,
            "standardization_npz": final_std_npz,
            "training_curve_csv": final_training_curve_csv,
        }
        with open(final_metrics_json, "w", encoding="utf-8") as f:
            json.dump(final_row, f, indent=2, ensure_ascii=False)
        _close_dataset(final_ds)
        del final_loader, trainer, optimizer, scheduler, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    export_best_ckpt, export_last_ckpt, export_stage_name, export_fold_id = select_best_fold_from_metrics(
        cfg,
        stage_name="TSCV",
    )
    if os.path.exists(export_best_ckpt):
        load_path = export_best_ckpt
    elif os.path.exists(export_last_ckpt):
        load_path = export_last_ckpt
    else:
        raise FileNotFoundError(
            f"No {export_stage_name} checkpoint found:\n"
            f"  best: {export_best_ckpt}\n"
            f"  last: {export_last_ckpt}"
        )

    print(f"[LOAD EXPORT CKPT] source={export_stage_name} fold={export_fold_id} path={load_path}")
    best = torch.load(load_path, map_location=cfg["DEVICE"])
    ref_ds = MultiSourceDataFusionDataset(
        stack_dir=cfg["STACK_DIR"],
        fine_label_dir=cfg["FINE_LABEL_DIR"],
        coarse_label_dir=cfg["COARSE_LABEL_DIR"],
        yyyymm_list=[final_months[-1]],
        tile_size=cfg["TILE_SIZE"],
        stride=cfg["STRIDE"],
        t_window=cfg["T_WINDOW"],
        nodata=cfg["NODATA"],
        use_coarse_loss=True,
        transform=None,
        min_valid_frac=0.0,
    )
    X0, _, _, _, _, _, _, _ = ref_ds[0]
    _, C, _, _ = X0.shape
    ref_ds.close()

    model = _make_model(cfg, C)
    model.load_state_dict(best["model_state"])
    model.eval()
    os.makedirs(export_folder, exist_ok=True)

    for yyyymm in export_months:
        out_tif = build_ehdi_export_path(export_folder, cfg, export_stage_name, export_fold_id, yyyymm)
        if os.path.exists(out_tif):
            print(f"[RESUME] FINAL EXPORT exists, skip: {out_tif}")
            continue
        export_ds = MultiSourceDataFusionDataset(
            stack_dir=cfg["STACK_DIR"],
            fine_label_dir=cfg["FINE_LABEL_DIR"],
            coarse_label_dir=cfg["COARSE_LABEL_DIR"],
            yyyymm_list=[yyyymm],
            tile_size=cfg["TILE_SIZE"],
            stride=cfg["STRIDE"],
            t_window=cfg["T_WINDOW"],
            nodata=cfg["NODATA"],
            use_coarse_loss=True,
            transform=Standardizer(best["mean"], best["std"], cfg["NODATA"]),
            min_valid_frac=0.0,
        )
        export_fused_images(
            model=model,
            dataset=export_ds,
            output_folder=export_folder,
            yyyymm=yyyymm,
            device=cfg["DEVICE"],
            nodata_value=cfg["NODATA"],
            dtype="float32",
            cfg=cfg,
            stage_name=export_stage_name,
            fold_id=export_fold_id,
        )
        export_ds.close()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[MS-TCFNet DONE] out_dir={cfg['OUT_DIR']}\n")
