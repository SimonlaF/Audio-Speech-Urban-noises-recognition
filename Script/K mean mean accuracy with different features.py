import librosa
import numpy as np
import os
import random # Ensure random is imported for the shuffling logic
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import statistics
from statistics import mean 
import matplotlib.pyplot as plt

# Defining paths
path_background = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\audio_filtered'
path_speech = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\LibriSpeech\\audio_total'

# Store accuracies for multiple runs
accuracy = []

# Extract features (Clean Background)
def get_features(folder, extension, label_val, n_samples=1000):
    all_files = [f for f in os.listdir(folder) if f.endswith(extension)]
    
    # Randomly shuffle and select files to ensure diverse data selection
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
        
    
        # Feature 2: Various options for second feature extraction

        # Spectral centroid extraction
        #spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        #feat2 = np.mean(spectral_centroids)

        # Spectral Rolloff extraction 
        #spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.90)[0]
        #feat2 = np.mean(spectral_rolloff)

        #ZCR Extraction 
        #zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        #feat2 = np.mean(zcr)
        
        # Energy extraction 
        energy = librosa.feature.rms(y=y)[0]
        feat2 = np.std(energy) / np.mean(energy)
        
        features.append([feat1, feat2])            
    return np.array(features), [label_val] * len(features)

# Loop for executing 100 differents K-means
for f in range(100): 
    
    # Data Extraction
    X_back, y_b = get_features(path_background, "wav", 0, 1000)
    X_speech, y_s = get_features(path_speech, "flac", 1, 1000)

    X = np.vstack([X_back, X_speech])
    y_true = np.concatenate([y_b, y_s])

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K means clustering
    K = 2
    n_iterations = 5
    # Random initialization of centroids
    centroids = X_scaled[np.random.choice(len(X_scaled), K, replace=False)]
 
    # K-means clustering loop
    for i in range(n_iterations):
        # Calculate Euclidean distances to centroids
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids by calculating the mean of points in each cluster
        new_centroids = np.array([X_scaled[labels == k].mean(axis=0) for k in range(K)])
        centroids = new_centroids


    # Comparing K-means clusters with real classes
    # K-means assigns arbitrary cluster IDs, so we check both mapping possibilities
    acc1 = accuracy_score(y_true, labels)
    acc2 = accuracy_score(y_true, 1 - labels)
    final_accuracy = max(acc1, acc2)
    print(final_accuracy)
    accuracy.append(final_accuracy)
print(mean(accuracy)*100)