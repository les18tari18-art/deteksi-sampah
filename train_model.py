"""
Training model klasifikasi sampah Organik vs Anorganik (Transfer Learning MobileNetV2).

Dataset: Kaggle "Waste Classification Data" (Sashaank Sekar)
Struktur folder yang dibutuhkan:
    DATASET/
        TRAIN/
            O/   (organik)
            R/   (recyclable / anorganik)
        TEST/
            O/
            R/
Jalankan di Google Colab (GPU) atau lokal:  python train_model.py
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "DATASET"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/TRAIN", validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/TRAIN", validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/TEST", image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

class_names = train_ds.class_names
print("Kelas:", class_names)
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Evaluasi pada data test
loss, acc = model.evaluate(test_ds)
print(f"\nAkurasi data test: {acc*100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = (model.predict(test_ds).ravel() > 0.5).astype(int)
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

# Grafik training
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Akurasi"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.savefig("grafik_training.png", dpi=150, bbox_inches="tight")

model.save("waste_model.keras")
print("Model tersimpan: waste_model.keras")
