# Research Study: Diffusion Models for Unsupervised Anomaly Detection in Fetal Brain Ultrasound

## 1. Overview

This directory documents the research paper **“Diffusion Models for Unsupervised Anomaly Detection in Fetal Brain Ultrasound”** by Hanna Mykula, Lisa Gasser, Silvia Lobmaier, Julia A. Schnabel, Veronika Zimmer, and Cosmin I. Bercea.

- **Research domain:** Medical Image Analysis, Deep Learning, Generative AI
- **Imaging modality:** Fetal brain ultrasound
- **Learning paradigm:** Unsupervised Anomaly Detection (UAD)
- **Primary models:** AnoDDPM and AutoDDPM
- **Paper:** https://arxiv.org/abs/2407.15119
- **Publication record:** arXiv preprint, July 2024
- **Project status:** Research analysis and implementation in progress

## 2. Research Problem

Fetal brain ultrasound is widely used to monitor fetal development. However, ultrasound images can contain speckle noise, acoustic shadows, motion artifacts, and variations in image quality.

Conventional supervised learning approaches require labeled examples of normal and abnormal cases. Obtaining sufficient labeled pathological images can be difficult, particularly when abnormalities are rare or diverse.

The paper addresses this challenge through unsupervised anomaly detection. The model learns patterns from healthy fetal brain ultrasound images and attempts to identify deviations from the learned normal appearance.

## 3. Research Objectives

1. Develop a preprocessing pipeline for selecting suitable fetal brain ultrasound images.
2. Identify the transventricular fetal brain standard plane.
3. Segment and crop the fetal brain region to reduce background interference.
4. Investigate diffusion-based methods for reconstructing pseudo-healthy images.
5. Compare AnoDDPM and AutoDDPM using different noise types and noise levels.
6. Evaluate image reconstruction quality and anomaly detection performance.

## 4. Methodology

The original framework consists of the following stages:

### Stage 1: Data Selection

Filter ultrasound images according to gestational age and image suitability.

The study selects cases within the 19 to 22+6 weeks gestational-age range for its clinical dataset.

### Stage 2: Standard Plane Detection

SonoNet identifies fetal ultrasound standard planes. The study selects images classified as the transventricular brain plane and applies a confidence threshold of 0.9.

### Stage 3: Brain Segmentation

A U-Net-based segmentation approach identifies the fetal head region. The selected images are cropped to reduce background interference, followed by manual quality review.

### Stage 4: Image Preprocessing

The paper resizes images to 128 × 128 pixels, normalizes pixel values, and applies rotation and horizontal and vertical flip augmentation.

### Stage 5: Diffusion-Based Reconstruction

The study evaluates:

- **AnoDDPM with Gaussian noise**
- **AnoDDPM with Simplex noise**
- **AutoDDPM with Gaussian noise**

The diffusion models reconstruct images from corrupted inputs. Differences between original and reconstructed images are used to identify potential anomalous regions.

### Stage 6: Anomaly Map Generation

The paper investigates reconstruction-based anomaly maps using Mean Absolute Error (MAE), Learned Perceptual Image Patch Similarity (LPIPS), and their combination.

### Stage 7: Evaluation

The study assesses reconstruction quality using MAE, SSIM, and LPIPS, and anomaly detection performance using Area Under the Precision-Recall Curve (AUPRC) and Area Under the Receiver Operating Characteristic Curve (AUROC).

## 5. Dataset Information

The paper reports using public and private clinical datasets.

| Dataset characteristic | Reported value |
|---|---:|
| Normal training/validation/testing pool | 252 images |
| Patients contributing to this pool | 95 |
| Anomalous evaluation patients | 8 |
| Anomalous evaluation images | 18 |
| Healthy evaluation patients | 5 |
| Healthy evaluation images | 12 |

The 252-image pool consists of data from 76 clinical patients and 19 patients from a public dataset. The separate downstream evaluation includes 30 images.

These figures describe the original study, not data currently included in this repository. The private clinical data must not be assumed to be publicly accessible.

## 6. Reported Experimental Results

The following results are taken from the original paper's combined MAE × LPIPS anomaly-map evaluation.

| Method | Noise | Level (t) | AUPRC (%) | AUROC (%) |
|---|---|---:|---:|---:|
| AnoDDPM | Gaussian | 250 | 73.0 | 63.8 |
| AnoDDPM | Gaussian | 300 | 73.5 | 57.4 |
| AnoDDPM | Simplex | 50 | 78.9 | 70.8 |
| AutoDDPM | Gaussian | 200 | 77.6 | 65.7 |
| AutoDDPM | Gaussian | 300 | 79.8 | 66.6 |

**Key finding:** AutoDDPM with Gaussian noise at t = 300 achieved the highest reported AUPRC of 79.8%. AnoDDPM with Simplex noise at t = 50 achieved the highest reported AUROC of 70.8%.

These results illustrate the importance of evaluating multiple metrics. The method with the best precision-recall performance is not necessarily the method with the best ROC performance.

## 7. Limitations Identified in the Paper

- The anomaly evaluation dataset is small.
- More extensive data is needed for stronger statistical validation.
- Simplex noise can introduce reconstruction artifacts.
- Reconstruction quality and anomaly detection performance involve a trade-off.
- False-positive anomaly regions remain a challenge.
- Generalization across ultrasound machines and acquisition settings requires further investigation.
- Evaluation on ultrasound video data remains a future research direction.

## 8. Proposed Extensions for This Project

The following are proposed experiments, not established improvements. Each should be implemented and evaluated before drawing conclusions.

1. **Systematic noise-level optimization:** Compare a broader range of diffusion noise levels and select operating points using validation data.
2. **Anomaly-map comparison:** Evaluate MAE, LPIPS, and combined anomaly maps under a consistent evaluation protocol.
3. **Robustness evaluation:** Investigate sensitivity to speckle noise, image quality, and acquisition variations.
4. **Preprocessing ablation:** Measure the contribution of plane detection, segmentation, and image normalization.
5. **Threshold calibration:** Select anomaly-score thresholds using validation data and report sensitivity, specificity, and false-positive rates.
6. **Reproducibility:** Record random seeds, configuration files, model checkpoints, software versions, and evaluation settings.
7. **External validation:** If suitable, authorized datasets become available, evaluate performance across independent cohorts and ultrasound acquisition settings.

## 9. Planned Repository Components

- `original_research_paper.pdf` — the source paper
- `research_summary.md` — concise paper summary
- `methodology.md` — detailed explanation of the pipeline
- `limitations_and_research_gaps.md` — limitations and proposed extensions
- `experiments/` — experiment configurations and evaluation records
- `figures/` — methodology and results visualizations
- `references.bib` — bibliographic references

Implementation status and experimental results should be updated as the project progresses.

## 10. Research Disclaimer

This repository is an academic research project exploring unsupervised anomaly detection in fetal brain ultrasound. The described methods and reported results are not evidence of clinical readiness.

The project is not a medical device and must not be used to diagnose, rule out, or make clinical decisions about fetal abnormalities.

## 11. Citation

Mykula, H., Gasser, L., Lobmaier, S., Schnabel, J. A., Zimmer, V., & Bercea, C. I. (2024). *Diffusion Models for Unsupervised Anomaly Detection in Fetal Brain Ultrasound*. arXiv:2407.15119.

https://doi.org/10.48550/arXiv.2407.15119
