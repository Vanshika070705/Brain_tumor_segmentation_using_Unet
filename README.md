# Brain Tumor Segmentation using U-Net with Zoom-on-Mask Augmentation

## Overview

This repository presents a deep learning pipeline for **brain tumor segmentation** using a modified **U-Net architecture**. The primary innovation is a custom **"zoom-on-mask" data augmentation** strategy and a **fail-safe inference mechanism** to improve segmentation of small tumor regions.

The model is trained to segment tumor regions from brain MRI images, outputting binary masks indicating tumor locations.

---

## Features

* **Enhanced U-Net Architecture**: Incorporates attention gates for improved segmentation precision.
* **Zoom-on-Mask Augmentation**:

  * Dynamically zooms into the tumor region during training.
  * Helps the model learn fine-grained features of small tumors.
* **Custom Weighted Loss Function**:

  * **Weighted Dice Loss**: Emphasizes tumor pixels (foreground) more than background.
  * **Tversky Loss ($\beta = 0.7$)**: Penalizes false positives heavily.
  * **Focal Loss**: Focuses on hard-to-classify pixels.
* **Fail-Safe Inference Strategy**:

  * First predicts on a zoomed crop.
  * Falls back to full-image prediction if the crop yields no mask.
* **Colab-Ready**: Optimized for Google Colab with GPU support.
* **COCO Dataset Integration**: Uses `pycocotools` for dataset annotation and loading.

---

## Dataset

The model uses a COCO-style segmentation dataset with images and masks. Expected path structure:

```
/content/drive/MyDrive/BrainTumorSeg/tumor_dataset/test
```

You must provide your own brain tumor segmentation dataset in **COCO format**.

---

## Setup and Running

### 1. Clone the Repository

```bash
!git clone https://github.com/Vanshika070705/Brain_tumor_segmentation_using_Unet.git
%cd Brain_tumor_segmentation_using_Unet
```

### 2. Prepare Your Data

* **Upload `kaggle.json`** (if using Kaggle dataset):

```python
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json
```

* **Place your dataset**:

  * Upload brain tumor images and `*_annotations.coco.json`


* Adjust the `ANNOTATIONS_FILE` and `IMAGES_DIR` variables in the notebook if your dataset path differs.

### 3. Run the Colab Notebook

Open and execute all cells in the notebook:

```text
zoom_mask_final_git.ipynb
```

This includes:

* Google Drive mounting
* Kaggle API setup (optional)
* Dataset loader initialization
* Zoom-on-mask preprocessing
* tf.data pipeline
* U-Net model definition
* Custom loss and metrics
* Model training
* Prediction visualization

---

## Model Architecture

### U-Net with Attention Gates

* **Encoder**:

  * 3 Conv2D blocks + BatchNorm + ReLU + MaxPooling
  * Includes Dropout
* **Bottleneck**:

  * Deepest convolution block
* **Decoder**:

  * 3 UpSampling blocks + Conv2D + Attention Gates
  * Skip connections with encoder
* **Output**:

  * 1x1 Conv2D with sigmoid activation for binary output

---

## Loss Function

```math
Loss = 0.4 \times (1 - \text{Weighted Dice}) + 0.3 \times \text{Tversky Loss} + 0.3 \times \text{Focal Loss}
```

* **Weighted Dice Loss**:
  
  * $\text{weight-fg} = 10.0$, $\text{weight-bg} = 1.0$

* **Tversky Loss**:

  * $\alpha = 0.3, \beta = 0.7$ for higher false-positive penalty
* **Focal Loss**:

  * $\gamma = 2.0, \alpha = 0.25$


---

## Metrics

During training:

* **Dice Coefficient**
* **IoU Score (Intersection over Union)**

---

## Results Visualization

The utility function `visualize_zoom_with_failsafe` generates plots showing:

* Original MRI Image
* Ground Truth Mask
* Zoomed Crop Region
* Prediction on Zoomed Crop (Overlayed)
* Final Output on Full Image (with fallback indication)



## Author

[Vanshika Malik](https://github.com/Vanshika070705)

If you find this project helpful, consider starring the repository!
