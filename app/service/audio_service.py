import os
import joblib
import librosa
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class AudioService:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def extract_features(self, file_path: str, sample_rate: int = None, augment: bool = False, is_real: bool = False) -> np.ndarray:
        try:
            # Load audio file directly with librosa
            audio, sr = librosa.load(file_path, sr=sample_rate)
            
            if augment:
                noise = np.random.normal(0, 0.015, audio.shape)
                audio = audio + noise
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-4, 4))
                if is_real:
                    audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.6, 1.4))
            
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            spectral_contrast_std = np.std(spectral_contrast, axis=1)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)

            features = np.concatenate([
                mfcc_mean, mfcc_std,
                spectral_contrast_mean, spectral_contrast_std,
                [zcr_mean, zcr_std]
            ])
            
            return features
        
        except Exception as e:
            raise ValueError(f"Error extracting features: {str(e)}")

    def predict(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        features = self.extract_features(file_path, augment=False, is_real=False)
        if features is None:
            raise ValueError(f"Could not extract features from {file_path}")
        
        features_scaled = self.scaler.transform([features])
        proba = self.model.predict_proba(features_scaled)[0, 1]
        prediction = 1 if proba >= 0.5 else 0
        
        return "Real audio" if prediction == 0 else "Fake audio" 