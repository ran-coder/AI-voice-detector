#using the MFCC features saved by extract_mfcc.py.

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


FEATURES_DIR = "features"

def load_data():
    X_train = np.load(f"{FEATURES_DIR}/X_train.npy")   
    y_train= np.load(f"{FEATURES_DIR}/y_train.npy")   # shape: (num_samples,) -- 1=bonafide, 0=spoof
    X_dev = np.load(f"{FEATURES_DIR}/X_dev.npy")   
    y_dev= np.load(f"{FEATURES_DIR}/y_dev.npy") 
    return X_train, y_train, X_dev, y_dev

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),   # binary output: bonafide vs spoof
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("Loading features...")
    X_train, y_train, X_dev, y_dev = load_data()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape}")

    print(f"Class balance -- bonafide: {(y_train == 1).sum()}, spoof: {(y_train == 0).sum()}")

    X_train = X_train[..., np.newaxis]
    X_dev = X_dev[..., np.newaxis]

    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
 

    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=15,
        batch_size=32,
        class_weight=class_weight_dict
    )
    
    val_loss, val_acc = model.evaluate(X_dev, y_dev)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    y_pred = (model.predict(X_dev) > 0.5).astype(int).flatten()

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print("             pred_spoof  pred_bonafide")
    cm = confusion_matrix(y_dev, y_pred)
    print(f"actual_spoof     {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"actual_bonafide  {cm[1][0]:>6}      {cm[1][1]:>6}")
 
    print("\nClassification report:")
    print(classification_report(y_dev, y_pred, target_names=["spoof", "bonafide"]))
 
    model.save("voice_authenticity_cnn.keras")
    print("Model saved to voice_authenticity_cnn.keras")
 
if __name__ == "__main__":
    main()
