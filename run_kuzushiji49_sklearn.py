### vv chat gpt generated


import numpy as np
from keras import layers, models
from sklearn.model_selection import train_test_split

X = np.load("data/kuzushiji-49/train_imgs.npz")["arr_0"]
y = np.load("data/kuzushiji-49/train_labels.npz")["arr_0"]

print("Loaded KMNIST-49 dataset:", X.shape, y.shape)

# Normalize (0–255 -> 0–1 float)
X = X.astype("float32") / 255.0

# Flatten 28x28 images -> 784-length vectors
X = X.reshape(-1, 28 * 28)

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 2️⃣ Build the model
# -------------------------------
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),   # Hidden layer 1
    layers.Dense(128, activation='relu'),                       # Hidden layer 2
    layers.Dense(49, activation='softmax')                      # Output layer (49 classes)
])

# -------------------------------
# 3️⃣ Compile the model
# -------------------------------
model.compile(
    optimizer='sgd',                          # Optimizer adjusts weights
    loss='sparse_categorical_crossentropy',    # For integer class labels
    metrics=['accuracy']                       # Track accuracy during training
)

model.summary()  # Print model structure

# -------------------------------
# 4️⃣ Train the model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=10, batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -------------------------------
# 5️⃣ Evaluate the model
# -------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# -------------------------------
# 6️⃣ Inspect weights (optional)
# -------------------------------
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f"\nLayer {i}: {layer.name}")
    print(f" - Weights shape: {weights.shape}")
    print(f" - Biases shape:  {biases.shape}")
