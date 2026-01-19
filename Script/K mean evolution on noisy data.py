import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn

# Defining paths
path_background = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\audio_filtered'
path_speech = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\LibriSpeech\\audio_total'

# Extract features (Clean Background)
def get_features(folder, extension, label_val, n_samples=1000):
    all_files = [f for f in os.listdir(folder) if f.endswith(extension)]
    
    # Randomly shuffle and select files
    random.shuffle(all_files)
    files = all_files[:n_samples]
    
    features = []
    for f in tqdm(files, desc=f"Loading Class {label_val}"):
        # Loading files
        path = os.path.join(folder, f)
        y, sr = librosa.load(path, sr=16000, duration=5.0)
        
        # Feature 1: MFCC 2 Standard Deviation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feat1 = np.std(mfcc[1, :])
        
        # Feature 2: Energy Ratio (Temporal Variation)
        energy = librosa.feature.rms(y=y)[0]
        ratio_energy = np.std(energy) / np.mean(energy)
        feat2 = ratio_energy
        
        features.append([feat1, feat2])         
    return np.array(features), [label_val] * len(features)

# Extract features (Noisy Speech: Mixing Voice + SONYC Noise)
def get_noisy_features(folder_speech, folder_back, ext_s, ext_b, n_samples, noise_factor):
    all_files_s = [f for f in os.listdir(folder_speech) if f.endswith(ext_s)]
    all_files_b = [f for f in os.listdir(folder_back) if f.endswith(ext_b)]
    
    # Shuffle lists to ensure random selection
    random.shuffle(all_files_s)
    random.shuffle(all_files_b)
    
    files_s = all_files_s[:n_samples]
    files_b = all_files_b[:n_samples]
    
    features = []
    for i in tqdm(range(len(files_s)), desc="Extracting Noisy Speech"):
        # Loading files
        y_s, sr = librosa.load(os.path.join(folder_speech, files_s[i]), sr=16000, duration=5.0)
        y_b, _ = librosa.load(os.path.join(folder_back, files_b[i]), sr=16000, duration=5.0)
        
        # Length alignment
        length = min(len(y_s), len(y_b))
        
        # Mixing : Voice + (Noise * noise_factor)
        y_mixed = y_s[:length] + noise_factor * y_b[:length]
        
        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y_mixed, sr=sr, n_mfcc=13)
        feat1 = np.std(mfcc[1, :])
        
        rms = librosa.feature.rms(y=y_mixed)[0]
        feat2 = np.std(rms) / np.mean(rms)
        
        features.append([feat1, feat2])
    return np.array(features), [1] * len(features)

# Execute Data Extraction
X_back, y_b = get_features(path_background, "wav", 0, 1000)
X_noisy_speech, y_ns = get_noisy_features(path_speech, path_background, "flac", "wav", 1000, noise_factor=2)
X = np.vstack([X_back, X_noisy_speech])
y_true = np.concatenate([y_b, y_ns])

# Normalization 
X_scaled = StandardScaler().fit_transform(X)

# K-means clustering
K = 2
n_iterations = 5

# Random centroid initialization
centroids = X_scaled[np.random.choice(len(X_scaled), K, replace=False)]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# K-means clustering loop
for i in range(n_iterations):
    # Calculate Euclidean distances to centroids
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    ax = axes[i]
    ax.scatter(X_scaled[labels==0, 0], X_scaled[labels==0, 1], c='blue', alpha=0.3, s=15, label='Cluster 0')
    ax.scatter(X_scaled[labels==1, 0], X_scaled[labels==1, 1], c='red', alpha=0.3, s=15, label='Cluster 1')
    ax.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='X', s=200, edgecolors='black', label='Centroids')
    ax.set_title(f"K-means - Iteration {i+1}")
    ax.set_xlabel("MFCC 2 Std")
    ax.set_ylabel("Energy")
    
    # Update centroids by calculating the mean of points in each cluster
    new_centroids = np.array([X_scaled[labels == k].mean(axis=0) for k in range(K)])
    centroids = new_centroids

# Plot ground truth
ax_real = axes[5]
ax_real.scatter(X_scaled[y_true==0, 0], X_scaled[y_true==0, 1], c='blue', alpha=0.3, s=15, label='Real Background')
ax_real.scatter(X_scaled[y_true==1, 0], X_scaled[y_true==1, 1], c='red', alpha=0.3, s=15, label='Real Speech')
ax_real.set_title("GROUND TRUTH (Real Labels)")
ax_real.set_xlabel("MFCC 2 Std")
ax_real.set_ylabel("Energy")
ax_real.legend()
plt.tight_layout()
plt.suptitle("K-means Evolution and Ground Truth (MFCC 2 & Energy Ratio)", fontsize=16, y=1.02)
plt.show()


# Comparing K-means clusters with real classes
# K-means assigns arbitrary cluster IDs, so we check both mapping possibilities
acc1 = accuracy_score(y_true, labels)
acc2 = accuracy_score(y_true, 1 - labels)
final_accuracy = max(acc1, acc2)
final_labels = labels if acc1 > acc2 else 1 - labels

# Plotting the Confusion Matrix to visualize classification performance
cm = confusion_matrix(y_true, final_labels)
plt.figure(figsize=(8, 6))
seaborn.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Engine (SONYC)', 'Voice (Libri)'],
                yticklabels=['Engine (SONYC)', 'Voice (Libri)'])
plt.title(f"K-Means Confusion Matrix\nAccuracy: {final_accuracy:.2%}")   
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.show()