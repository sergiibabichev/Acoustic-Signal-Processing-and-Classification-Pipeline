# Acoustic-Signal-Processing-and-Classification-Pipeline
This repository contains scripts for processing, filtering, visualizing, and classifying experimental acoustic signals stored in `.wav` format. The workflow consists of multiple steps, including data transformation, empirical mode decomposition (EMD), wavelet filtering, and classification.

## Files and Their Functions

### **1. Data Loading and Transformation**
- **`data_loading_transforming.py`**  
  Reads and transforms `.wav` files into a numeric format.  
  **Output:** Five `.csv` files saved in the working directory.

### **2. Signal Visualization**
- **`signals_visualisation.py`**  
  Visualizes the original signals.

### **3. Empirical Mode Decomposition (EMD)**
- **`EMD_Transform_functions.py`**  
  Performs EMD and extracts intrinsic mode functions (IMFs).  
  **Output:** Five `.csv` files containing time and IMFs.

### **4. IMF Visualization**
- **`IMFs_Visualization.py`**  
  Visualizes IMFs vs. time for **Signal 1**.

### **5. Criteria Calculation**
- **`criteria_functions.py`**  
  Contains functions for calculating **Signal-to-Noise Ratio (SNR)** and **SURE criteria**.

### **6. Wavelet Analysis**
- **`wavelet_functions.py`**  
- **`main_filtering.py`**  
  Perform a stepwise procedure for wavelet analysis.

### **7. IMFs Filtering and Signal Reconstruction**
- **`Pipline_wavelet_functions.py`**  
- **`main_pipline_filtering.py`**  
  Provide a pipeline for filtering IMFs and reconstructing signals.

### **8. Filtration Results Visualization**
- **`filtration_Results_Visualization.py`**  
  Visualizes the results of the filtration process.

### **9. Signal Classification**
- **`signals_classification_functions.py`**  
- **`signals_classification_main.py`**  
  Perform the classification procedure and visualize classification results.

## How to Use
1. Place your `.wav` and .py files in the working directory.
2. Run `data_loading_transforming.py` to convert signals to numerical format.
3. Use `signals_visualisation.py` to inspect the original signals.
4. Apply `EMD_Transform_functions.py` to extract IMFs.
5. Visualize IMFs with `IMFs_Visualization.py`.
6. Perform wavelet filtering applying `main_filtering.py`.
8. Use the pipeline scripts to filter IMFs and reconstruct signals (Apply main_pipline_filtering.py).
9. Finally, execute `signals_classification_main.py` to classify signals.


