import os
import joblib
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utilities.build_data import prepare_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ravdess_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAVDESS_Audio_Files'))
cremad_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Crema-D_Audio_Files'))
tess_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TESS_Audio_Files'))

X, y = prepare_data(
    ravdess_path=ravdess_path,
    cremad_path=cremad_path,
    tess_path=tess_path,
)


X = np.where(np.isinf(X), np.nan, X)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
_, counts = np.unique(y_encoded, return_counts=True)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LDA()),
    ('svc', SVC())
])

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipe, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=3)

print("🚀 Starting model training...")
grid.fit(X, y_encoded)

print("\n✅ Training complete.")
print("CV accuracy with best parameters:", grid.best_score_)
print("Best parameters:", grid.best_params_)

y_pred = grid.predict(X)
class_names = encoder.classes_
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

model_path = 'packages/emotion_lda_svm_pipeline.joblib'
encoder_path = 'packages/label_encoder.joblib'
os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(grid.best_estimator_, model_path)
joblib.dump(encoder, encoder_path)