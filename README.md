# Long-Term Vegetation and Landscape Change Analysis in Bangalore

## Project Overview

This project analyzes long-term vegetation and landscape changes in Bangalore using Google Earth Engine (GEE) and a suite of advanced machine learning and deep learning models.

While initial baselines were established using Random Forest, the project evolved to explore **Deep Learning architectures (CNNs and LSTMs)** to address specific challenges in distinguishing spectrally similar classes (e.g., Cropland vs. Sparse Vegetation) in complex urban-rural transition zones.

## Datasets

The following datasets are utilized in this analysis:

| Dataset | Description | Source |
| --- | --- | --- |
| **Sentinel-2 MSI (Level-2A)** | Harmonized satellite imagery used for both static composites and monthly time-series generation. | [Catalog Link](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| **Bengaluru Shapefile** | Provides the Area of Interest (AOI) for the study. | [KSRSAC Link](https://kgis.ksrsac.in/bengalurugis/) |

---

## Methodology

The project adopted a multi-phase experimental approach to improve classification accuracy.

### Phase 1: Static Feature Engineering (RF & SVM)

A 2020 median composite was used to derive spectral and textural features.

* **Input:** Single Median Composite (2020).
* **Features:** Raw Bands (B2-B12), Indices (NDVI, NDBI, NDWI), Topography (Slope/Elevation), and Texture (GLCM).
* **Models:** Random Forest (RF), Support Vector Machine (SVM - RBF Kernel), Gradient Boosting (XGBoost/LightGBM).

### Phase 2: Spatial & Spectral Deep Learning (CNNs)

To capture spatial context and complex spectral relationships, Convolutional Neural Networks were implemented.

* **1D-CNN (Spectral):** Treated the spectral bands of a single pixel as a 1D sequence to learn non-linear spectral signatures.
* **2D-CNN (Spatial):** Extracted **3x3** and **15x15** image patches to classify the center pixel based on neighborhood texture and shape context.
* **Data Pipeline:** Custom extraction to **TFRecords** to handle large patch datasets efficiently.

### Phase 3: Temporal Deep Learning (LSTM)

*Diagnosis:* The static models (Phase 1 & 2) struggled to distinguish **Cropland** from **Sparse Vegetation** because they look identical in a single median image.

* **Solution:** A Recurrent Neural Network (LSTM) was implemented to analyze the **Phenology** (growth cycle) of vegetation.
* **Input:** 12-Month Time Series of NDVI (Jan–Dec 2020).
* **Architecture:** Long Short-Term Memory (LSTM) network to classify pixels based on their temporal evolution (e.g., distinguishing the high-variance growth cycle of crops from the stable signal of forests).

---

## Model Architectures & Experimental Results

### 1. Comparative Leaderboard

The table below summarizes the performance of the various models tested.

| Model Architecture | Input Type | Overall Accuracy | Kappa Coefficient | Key Observation |
| --- | --- | --- | --- | --- |
| **K-Means Clustering** | Unsupervised | **78.31%** | 0.72 | *Baseline benchmark (Likely overfitted to spatial clusters).* |
| **Random Forest (RF)** | Static (Pixel) | **67.65%** | 0.60 | *The reliable standard. robust to noise but limited by spectral confusion.* |
| **1D-CNN** | Static (Pixel) | **65.88%** | 0.57 | *Struggled with limited data; performed worse than RF.* |
| **2D-CNN (3x3)** | Static (Patch) | **60.85%** | 0.51 | *Failed due to small patch size (10m res) and lack of distinct shapes.* |
| **LSTM (Time-Series)** | **Temporal (12-Month)** | *Evaluation In Progress* | -- | *Designed to solve the vegetation confusion via phenology.* |

### 2. Confusion Analysis

The primary challenge across all static models (RF, CNN) was the confusion between **Class 2 (Cropland)** and **Class 3 (Sparse Veg)**.

* **Reason:** Without temporal data, "Fallow Cropland" and "Scrubland" have nearly identical spectral signatures.
* **Correction:** The LSTM model addresses this by identifying the *change* in greenness: Cropland has a high variance (Brown \to Green \to Brown), while Sparse Veg remains relatively constant.

---

## Setup and Usage

This project uses a hybrid **Google Earth Engine (GEE)** + **TensorFlow** workflow.

### Prerequisites

```bash
pip install earthengine-api geemap tensorflow pandas matplotlib rasterio

```

### Execution Workflow

1. **Data Extraction (GEE):**
* Run the extraction scripts to generate training data.
* *Static Models:* Exports a CSV of spectral/textural features.
* *CNNs:* Exports **TFRecords** of image patches (3x3 or 15x15).
* *LSTM:* Exports a CSV of **12-month NDVI time series**.


2. **Model Training (Local/Colab):**
* Load data using `pandas` (for CSV) or `tf.data.TFRecordDataset` (for patches).
* Train the specific Keras model (defined in the `Models` section of the notebook).


3. **Inference & Mapping:**
* The notebook downloads a "Preview Region" (8x8 km).
* The trained model runs sliding-window inference on the local image.
* Results are visualized using `matplotlib` and exported as GeoTIFFs.



### Visualizing Results

The final output maps use the following color schema:

* **Water:** Blue (`#2c7fb8`)
* **Built/Barren:** Yellow (`#ffffcc`)
* **Cropland:** Light Green (`#c2e699`)
* **Sparse Veg:** Medium Green (`#31a354`)
* **Dense Veg:** Dark Green (`#006837`)