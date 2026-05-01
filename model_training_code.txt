# ✅ Imports
import os, cv2, json
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ✅ Paths
DATASET_DIR = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
LABELS_JSON_PATH = "class_indices_normalized.json"
MODEL_SAVE_PATH = "asl_mlp_normalized.keras"
PRETRAINED_MODEL_PATH = "/kaggle/working/asl_mlp_normalized.keras"  # Optional for continued training

# ✅ Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

X, y = [], []
class_names = sorted(os.listdir(DATASET_DIR))

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist  # Make wrist origin
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val > 0:
        landmarks /= max_val  # Normalize scale
    return landmarks.flatten()

# ✅ Extract and Normalize
for label in class_names:
    for img_name in tqdm(os.listdir(os.path.join(DATASET_DIR, label)), desc=f"Processing {label}"):
        img_path = os.path.join(DATASET_DIR, label, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        if result.multi_hand_landmarks:
            lm = []
            for pt in result.multi_hand_landmarks[0].landmark:
                lm.extend([pt.x, pt.y, pt.z])
            if len(lm) == 63:
                X.append(normalize_landmarks(lm))
                y.append(label)
hands.close()
X = np.array(X)

# ✅ Encode Labels
le = LabelEncoder()
y_cat = to_categorical(le.fit_transform(y))
with open(LABELS_JSON_PATH, 'w') as f:
    json.dump({i: label for i, label in enumerate(le.classes_)}, f)

# ✅ Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.1, random_state=42)

# ✅ Build/Load Model
if os.path.exists(PRETRAINED_MODEL_PATH):
    print("✅ Loading pretrained model...")
    model = load_model(PRETRAINED_MODEL_PATH)
else:
    print("🆕 Creating new model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20, batch_size=64,
                    callbacks=callbacks)

# ✅ Save
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# ✅ Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss"); plt.legend()
plt.tight_layout(); plt.show()