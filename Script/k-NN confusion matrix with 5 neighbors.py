import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import random

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
        
        # Feature extraction on noisy audio
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

# Splitting data between train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true, test_size=0.2)

# Training
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Prediction and score
y_pred = knn_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

#Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Background (SONYC)', 'Voice (Libri)'],
            yticklabels=['Background (SONYC)', 'Voice (Libri)'])
plt.title(f"Accuracy: {acc:.2%}")
plt.show()