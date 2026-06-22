## MS-TCFNet

### Model Architecture

MS-TCFNet is a multi-scale supervised spatiotemporal deep learning network designed for 30 m ecohydrological drought reconstruction under coarse meteorological supervision. The model integrates sequential spatial encoding, temporal-context learning, feature-wise modulation, and skip-connected spatial decoding to generate aggregation-consistent EHDI maps from monthly predictor sequences.

The architecture consists of four core components:

**ConvLSTM Encoder** processes monthly predictor stacks and extracts sequential spatial features while preserving temporal dependence across antecedent months. This module enables the model to incorporate short- to medium-term ecohydrological memory from meteorological forcing, vegetation condition, surface moisture, wetness, and land-surface thermal status.

**Temporal Transformer Block** models dependencies among monthly representations using self-attention. It allows the model to adaptively weight information from different antecedent months, rather than treating each monthly input as an independent observation.

**FiLM Modulation Module** converts the learned temporal context into feature-wise scale and shift parameters. These parameters modulate the final ConvLSTM hidden representation, allowing temporal context to condition the spatial decoding process.

**U-Net Decoder** reconstructs the 30 m EHDI field through skip-connected spatial decoding. The decoder helps preserve local spatial gradients and fine-resolution structure while producing continuous drought estimates.

In addition, MS-TCFNet is trained with a multi-scale supervised objective that combines pixel-level consistency, structural regularization, and aggregation-level consistency. This design constrains the model output at both the 30 m supervisory scale and the original 1 km meteorological scale, improving local spatial organization while maintaining consistency with the coarse drought reference.
