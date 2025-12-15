# ============================================
# FACE TRAINING USING CNN (FINAL FIX)
# ============================================

import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ================================
# 1️⃣ SEED (BIAR STABIL)
# ================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ================================
# 2️⃣ PATH & PARAMETER
# ================================
train_dir = 'dataset/train'
test_dir  = 'dataset/test'
img_size  = (150, 150)
batch_size = 32
epochs = 40

# ================================
# 3️⃣ DATA AUGMENTATION
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False   # 🔥 WAJIB
)

# ================================
# 4️⃣ SIMPAN LABEL MAPPING
# ================================
np.save("class_indices.npy", train_data.class_indices)
print("✔ Label mapping disimpan:", train_data.class_indices)

# ================================
# 5️⃣ CLASS WEIGHT (BIAR ADIL)
# ================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("✔ Class Weight:", class_weights)

# ================================
# 6️⃣ MODEL CNN
# ================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 7️⃣ CALLBACKS
# ================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )
]

# ================================
# 8️⃣ TRAINING
# ================================
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================================
# 9️⃣ EVALUASI BENAR
# ================================
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\n📊 CLASSIFICATION REPORT")
print(
    classification_report(
        test_data.classes,
        y_pred,
        target_names=list(test_data.class_indices.keys())
    )
)

print("\n📊 CONFUSION MATRIX")
print(confusion_matrix(test_data.classes, y_pred))

# ================================
# 🔟 SIMPAN MODEL
# ================================
model.save('face_cnn_model.h5')
print("\n✅ Model CNN FINAL berhasil disimpan")
