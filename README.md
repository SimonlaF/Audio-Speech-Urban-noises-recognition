# Audio Pattern Recognition: Speech vs. Urban Noise

## 1. Project Description
This project focuses on the automatic classification of human speech versus urban background noise. 

The system is designed to detect human voices in dense urban environments, this will allow wearable devices to switch automatically to "transparency mode" when someone speaks, ensuring the user remains connected to their social environment.

## 2. Methodology
The project follows a multi-domain feature extraction pipeline:

### Feature Extraction
Two low-complexity descriptors were chosen for their physical explainability:
* **Spectral Domain (MFCC 2):** Standard deviation of the 2nd Mel-Frequency Cepstral Coefficient to capture phonetic variance.
* **Temporal Domain (Energy Ratio):** The ratio of RMS deviation to total RMS energy, highlighting the "bursty" nature of speech compared to continuous noise.

### Dataset
* **Voice:** LibriSpeech dataset (training set of 100 hours "clean" speech) : [link](www.openslr.org/12) 
* **Noise:** SONYC Urban Sound Tagging dataset : [link](zenodo.org/records/3966543)
* **Preprocessing:** Speech samples were mixed with urban noise at a "noise factor" of 2 (approx. -6 dB SNR) to simulate high-stress real-world conditions.

## 3. Results and Comparison
We evaluated three different algorithmic approaches:

| Model | Classification Type | Mean Accuracy |
| :--- | :--- | :--- |
| **K-means** | Unsupervised | 83.00% |
| **k-NN (k=10)** | Supervised | 89.90% |
| **SVM (RBF Kernel)** | Supervised | **90.50%** |

**Key Findings:**
* **SVM with RBF Kernel** proved to be the optimal choice, offering the highest accuracy and the lowest memory footprint for embedded hardware.
* **Error Analysis:** Misclassifications were primarily caused by "speech-like" urban transients such as sirens and car horns.

## 4. Repository Structure
* `/scripts`: Python scripts for data filtering, feature extraction, and model benchmarking.
* `/report`: The final research paper in PDF format (generated with LaTeX).

## 5. Requirements
* Python 3.x
* `librosa`
* `scikit-learn`
* `numpy`
* `matplotlib`
* `seaborn`
* `tqdm`
* `pandas`

## 6. Author
* **Simon Ferrier**
