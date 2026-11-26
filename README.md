# 3D Gaussian Splatting Experiments: OOD Novel View Synthesis & Refinement

Empirical baselines, initialization comparisons, and diffusion post-processing ablations evaluating **3D Gaussian Splatting**, **MVSGaussian**, **SplatFormer**, and **Difix3D+** on out-of-distribution novel view synthesis (**OOD-NVS**).

## Overview
Sparse-view and out-of-distribution 3D scene reconstruction often suffers from geometric artifacts and incomplete point cloud initialization. This repository investigates a two-stage research pipeline on the **ShapeNet-OOD** benchmark:
1. **Geometry-Aware Initialization & Refinement:** Comparing pose-preserving COLMAP sparse triangulation against feed-forward multi-view stereo initialization (**MVSGaussian**) and eroded alpha-masked point cloud filtering (`_mvsinit_alpha`) prior to **Splatfacto** optimization and **SplatFormer** refinement.
2. **Diffusion Post-Processing Ablation:** Evaluating single-step diffusion restoration (**Difix3D+**) in both zero-shot pretrained (`nvidia/difix`) and domain-fine-tuned settings to analyze pixel-level fidelity vs. perceptual restoration on severe OOD geometric artifacts.

## Repository Structure

| Notebook | Description |
| :--- | :--- |
| [`MVSGaussain_+_Splatformer.ipynb`](MVSGaussain_+_Splatformer.ipynb) | End-to-end pipeline covering ShapeNet-OOD preparation, pose-preserving COLMAP triangulation, MVSGaussian `.ply` export, alpha-masked point filtering, Splatfacto training, SplatFormer refinement, and multi-view 3D point cloud evaluation. |
| [`Difix_ablation.ipynb`](Difix_ablation.ipynb) | Paired render-vs-GT dataset construction, resolution/weight-loading patches for `nvidia/difix`, 8-bit AdamW fine-tuning (`384×384`), and quantitative PSNR/SSIM/LPIPS ablation comparing pretrained vs. fine-tuned post-processing. |

## Key Engineering & Experimental Highlights

### 1. Alpha-Masked MVS Point Filtering (`_mvsinit_alpha`)
Raw feed-forward MVS depth fusion introduces floating background noise around object boundaries. Applying eroded alpha-channel masks before point cloud export significantly prunes background outliers while preserving core 3D bounding geometry:

| Scene ID | Raw MVS Points | Alpha-Filtered Points | Retention (%) | 3D BBox Diagonal |
| :--- | :---: | :---: | :---: | :---: |
| `02691156-2628b6...` | 5,627 | 1,928 | 34.3% | 1.23 |
| `02933112-d6138b...` | 809 | 141 | 17.4% | 2.12 |
| `02958343-1f6ce6...` | 1,839 | 1,082 | 58.8% | 1.61 |
| `02958343-9f703e...` | 922 | 241 | 26.1% | 1.81 |
| `03001627-1c199e...` | 1,319 | 381 | 28.9% | 1.88 |

### 2. Zero-Shot Diffusion Post-Processing Findings (`Difix3D+`)
* **Architectural Patches:** Resolved checkpoint loading mismatches (`pretrained_model_name_or_path` vs. `difix` pipeline initialization) and aligned training/inference spatial resolutions (`384×384`) for memory-efficient fine-tuning.
* **Quantitative Takeaway (10 OOD Scenes):** Zero-shot inference with `nvidia/difix` slightly decreases pixel-wise metrics (`PSNR: 12.44 → 12.33`, `SSIM: 0.6176 → 0.5439`, `LPIPS: 0.3636 → 0.3649`), demonstrating that single-step diffusion models hallucinate inconsistent textures when upstream 3D geometry is sparse—motivating task-specific fine-tuning on OOD render pairs.

## Tech Stack
`PyTorch` • `3D Gaussian Splatting` • `MVSGaussian` • `SplatFormer` • `Difix3D+` • `Nerfstudio (gsplat)` • `COLMAP` • `Diffusers / Accelerate`
