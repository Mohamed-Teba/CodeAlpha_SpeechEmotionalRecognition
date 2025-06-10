import joblib
from utilities.feature_extraction import extract_features

best_model_pipeline = joblib.load('../packages/emotion_lda_svm_pipeline.joblib')
label_encoder = joblib.load('../packages/label_encoder.joblib')
new_file_path = "../testcases/happy1.wav"
features = extract_features(new_file_path, duration=2.5, sample_rate=22050, top_db=20)
predicted_label_encoded = best_model_pipeline.predict([features])
predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
print(f"Predicted Emotion: {predicted_label[0]}")