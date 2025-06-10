import numpy as np
import librosa
import librosa.feature
from scipy.stats import skew, kurtosis


def preprocess_audio(file_path, duration=2.5, sample_rate=22050, top_db=20):
    y, _ = librosa.load(file_path, sr=sample_rate)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    target_length = int(sample_rate * duration)

    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')

    return y


def extract_features(file_path, duration=2.5, sample_rate=22050, top_db=20):
    y = librosa.util.normalize(preprocess_audio(file_path, duration, sample_rate, top_db))

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harmonic_rms = np.log1p(librosa.feature.rms(y=y_harmonic))
    percussive_rms = np.log1p(librosa.feature.rms(y=y_percussive))
    harmonic_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sample_rate)
    percussive_rolloff = librosa.feature.spectral_rolloff(y=y_percussive, sr=sample_rate)


    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
    rms = np.log1p(librosa.feature.rms(y=y))

    mel_spec = np.log1p(librosa.feature.melspectrogram(y=y, sr=sample_rate))
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)

    def stat_summary(x):
        return np.hstack([
            np.mean(x, axis=1),
            np.std(x, axis=1),
            skew(x, axis=1),
            kurtosis(x, axis=1),
            np.max(x, axis=1),
            np.median(x, axis=1)
        ])

    features = np.hstack([
        stat_summary(mfccs),
        stat_summary(mfccs_delta),
        stat_summary(mfccs_delta2),
        stat_summary(chroma),
        stat_summary(spectral_contrast),
        stat_summary(zero_crossing_rate),
        stat_summary(rolloff),
        stat_summary(rms),
        stat_summary(harmonic_rms),
        stat_summary(percussive_rms),
        stat_summary(harmonic_rolloff),
        stat_summary(percussive_rolloff),
        stat_summary(mel_spec),
        stat_summary(spectral_bandwidth)
    ])

    return features
