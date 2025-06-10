from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from utilities.build_ravdess import buildravdess
from utilities.build_cremad import buildcremad
from utilities.feature_extraction import extract_features
from utilities.build_tess import buildtess

def prepare_data(ravdess_path=None, cremad_path=None, tess_path=None, duration=2.5, sample_rate=22050, top_db=20):
    dfs = []
    info_msgs = []

    if ravdess_path:
        df_ravdess = buildravdess(ravdess_path)
        dfs.append(df_ravdess)
        info_msgs.append(f"[INFO] Added {len(df_ravdess)} RAVDESS files.")

    if cremad_path:
        df_cremad = buildcremad(cremad_path)
        dfs.append(df_cremad)
        info_msgs.append(f"[INFO] Added {len(df_cremad)} CREMA-D files.")

    if tess_path:
        df_tess = buildtess(tess_path)
        dfs.append(df_tess)
        info_msgs.append(f"[INFO] Added {len(df_tess)} TESS files.")

    if not dfs:
        raise ValueError("At least one dataset path must be provided.")

    for msg in info_msgs:
        print(msg)

    df = pd.concat(dfs, ignore_index=True)

    desired_emotions = {'neutral', 'happy', 'sad', 'angry'}
    assert set(df['emotion'].unique()).issubset(desired_emotions), "Unexpected emotion labels found!"

    min_count = df['emotion'].value_counts().min()
    df = df.groupby('emotion').apply(lambda x: x.sample(min_count, random_state=42))
    df.index = df.index.droplevel(0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    x, y = [], []
    for _, row in df.iterrows():
        features = extract_features(row['relative_path'], duration, sample_rate, top_db)
        x.append(features)
        y.append(row['emotion'])

    return np.array(x), np.array(y)