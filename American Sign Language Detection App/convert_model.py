import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Read weights directly from h5 file
weights_path = "keras_extracted/model.weights.h5"

with h5py.File(weights_path, 'r') as f:
    w0 = f['layers/dense/vars/0'][:]      # Dense(128) kernel
    b0 = f['layers/dense/vars/1'][:]      # Dense(128) bias
    w1 = f['layers/dense_1/vars/0'][:]    # Dense(64) kernel
    b1 = f['layers/dense_1/vars/1'][:]    # Dense(64) bias
    w2 = f['layers/dense_2/vars/0'][:]    # Dense(28) kernel
    b2 = f['layers/dense_2/vars/1'][:]    # Dense(28) bias

# Build model and set weights manually
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(28, activation='softmax')
])
model.build((None, 63))
model.layers[0].set_weights([w0, b0])
model.layers[2].set_weights([w1, b1])
model.layers[4].set_weights([w2, b2])

# Save in standard h5 format
model.save_weights("asl_weights.h5")
print("Done! asl_weights.h5 saved successfully.")