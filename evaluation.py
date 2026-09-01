'''done on the 2021 dataset'''

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

X_2021eval = np.load("features/X_2021eval.npy") # CNN was trained expecting a channel dimension
y_2021eval = np.load("features/y_2021eval.npy")

X_2021eval = X_2021eval[..., np.newaxis]

model = tf.keras.models.load_model("voice_authenticity_cnn.keras")

loss, acc = model.evaluate(X_2021eval, y_2021eval)
print(f"Accuracy on ASVspoof 2021 LA: {acc:.4f}")

y_pred = (model.predict(X_2021eval) > 0.5).astype(int).flatten()

print("\nConfusion matrix (rows=actual, cols=predicted):")
print("             pred_spoof  pred_bonafide")
cm = confusion_matrix(y_2021eval, y_pred)
print(f"actual_spoof     {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"actual_bonafide  {cm[1][0]:>6}      {cm[1][1]:>6}")

print("\nClassification report:")
print(classification_report(y_2021eval, y_pred, target_names=["spoof", "bonafide"]))