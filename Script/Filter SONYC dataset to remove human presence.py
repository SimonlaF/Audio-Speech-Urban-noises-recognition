import os
import shutil
import pandas as pd
from tqdm import tqdm

# Defining paths
source_base = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\SONYC\\audio'
dest_drive = 'C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\audio_filtered' 

os.makedirs(dest_drive, exist_ok=True)

# Loading annotations.csv
df = pd.read_csv('C:\\Users\\Simon\\Documents\\Cours\\Erasmus\\Audio pattern recognizion\\Project\\SONYC\\annotations.csv')

# Identify columns with human presence 
human_cols = [c for c in df.columns if c.startswith('7') and '_presence' in c]

# Keep lines where human presence = 0
mask_no_human = (df[human_cols] == 0).all(axis=1)
df_filtered = df[mask_no_human]
filenames = df_filtered['audio_filename'].unique()

# Copy loop 
for filename in tqdm(filenames):
    source_path = os.path.join(source_base, filename)
    
    if os.path.exists(source_path):
        dest_path = os.path.join(dest_drive, filename)
        shutil.copy2(source_path, dest_path)