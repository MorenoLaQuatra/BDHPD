
# BDHPD: Bilingual Dual-Head Architecture for Parkinson's Disease Detection from Speech

This repository contains the official implementation of the paper ["Bilingual Dual-Head Deep Model for Parkinson's Disease Detection from Speech"](link-to-paper), accepted at ICASSP 2025.

### Overview
BDHPD is a novel deep learning architecture designed for Parkinson's Disease detection from speech signals across multiple languages. The model employs:
- A dual-head architecture specialized for different speech tasks
- Self-supervised learning (SSL) and wavelet-based features
- Adaptive layers for cross-language generalization
- Convolutional bottlenecks for feature refinement
- Contrastive learning for enhanced discriminative capabilities

### Key Features
- Bilingual support (Slovak and Spanish)
- Specialized processing for diadochokinetic and continuous speech tasks
- Cross-language generalization capabilities

### Model Architecture
<p align="center">
  <img src="model_architecture.png" width="80%" alt="Model Architecture">
</p>

The model uses a shared backbone for feature extraction, followed by task-specific branches for PD detection.

### Repository Structure
```
TBD.
```

> [!IMPORTANT]  
> 🚧 This repository is currently being organized. Code, pre-trained models, and detailed documentation will be uploaded soon.

### Citation
If you use this code or find our work helpful, please cite:
```bibtex
@inproceedings{laquatra2025bilingual,
  title={Bilingual Dual-Head Deep Model for Parkinson's Disease Detection from Speech},
  author={La Quatra, Moreno and Orozco-Arroyave, Juan Rafael and Siniscalchi, Marco Sabato},
  booktitle={ICASSP 2025 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025}
}
```
