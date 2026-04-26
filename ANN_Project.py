# ANN Project — Iris Flower Classification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets       import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

#Load Dataset
iris       = load_iris()
X          = iris.data          # shape: (150, 4)  — 4 numerical features
y          = iris.target        # shape: (150,)    — 3 classes (0, 1, 2)
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

print("\nDataset Info:")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}  -> {list(iris.feature_names)}")
print(f"  Classes  : {list(class_names)}")
print(f"\nFirst 5 rows:")
for i in range(5):
    print(f"  {X[i]}  ->  class: {class_names[y[i]]}")

#Explore the Data
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Iris Dataset — Feature Distributions by Class', fontsize=14, fontweight='bold')

colors = ['#2196F3', '#FF5722', '#4CAF50']

for ax, feature_idx, name in zip(
    axes.flat,
    range(4),
    iris.feature_names
):
    for class_idx, (cls, color) in enumerate(zip(class_names, colors)):
        values = X[y == class_idx, feature_idx]
        ax.hist(values, bins=15, alpha=0.6, color=color, label=cls)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Value (cm)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: iris_01_feature_distributions.png")

#Split Data — Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing (30 samples), 80% for training (120 samples)
    random_state=42,    # fix randomness so results are reproducible
    stratify=y          # keep class ratio equal in both splits
)

print(f"\nTrain set : {X_train.shape[0]} samples")
print(f"Test  set : {X_test.shape[0]}  samples")

#Preprocessing — StandardScaler
# StandardScaler: transforms each feature to have mean=0 and std=1
# Formula: z = (x - mean) / std
# Why? Features have different scales (e.g. sepal length ~5cm vs petal width ~1cm)
# Without scaling the model may treat larger numbers as more important

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn mean/std from train, then scale
X_test  = scaler.transform(X_test)        # use SAME mean/std learned from train only

print(f"\nAfter scaling — Train set:")
print(f"  Mean : {X_train.mean(axis=0).round(4)}")
print(f"  Std  : {X_train.std(axis=0).round(4)}")

# Build ANN Model
model = keras.Sequential([

    # Input layer: 4 features
    layers.Input(shape=(4,)),

    # Hidden layer 1
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    # Hidden layer 2
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    # Output layer: 3 classes (softmax -> probabilities)
    layers.Dense(3, activation='softmax')
])

print("\n" + "="*50)
print("Model Architecture:")
print("="*50)
model.summary()

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train Model
# --------------------------------------------------
print("\nStarting training...")
print("-" * 50)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    verbose=1
)

print("\nTraining complete!")

#Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History — Iris ANN', fontsize=14, fontweight='bold')

ax1.plot(history.history['accuracy'],     label='Train Accuracy',     color='#2196F3', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF5722', linewidth=2, linestyle='--')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'],     label='Train Loss',     color='#4CAF50', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='#9C27B0', linewidth=2, linestyle='--')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_02_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: iris_02_training_curves.png")

#Evaluate on Test Set
print("\n" + "="*50)
print("Evaluation on Test Set:")
print("="*50)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\nSample Predictions (first 10 test samples):")
print(f"  {'True Label':<20} {'Predicted':<20} {'Confidence':>10}")
print("  " + "-"*52)
for i in range(10):
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]
    conf      = y_pred_probs[i][y_pred[i]] * 100
    status    = "OK" if y_test[i] == y_pred[i] else "WRONG"
    print(f"  {true_name:<20} {pred_name:<20} {conf:>8.1f}%   {status}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True, fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5
)
ax.set_title('Confusion Matrix — Iris ANN', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label',      fontsize=11)

plt.tight_layout()
plt.savefig('iris_03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: iris_03_confusion_matrix.png")

print("\nDetailed Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# Final Summary
print("\n" + "="*55)
print("Final Summary")
print("="*55)
print(f"  Dataset       : Iris Flower (150 samples, 4 features)")
print(f"  Model         : ANN  (4 -> 64 -> 32 -> 3)")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print("\nSaved plots:")
print("  iris_01_feature_distributions.png")
print("  iris_02_training_curves.png")
print("  iris_03_confusion_matrix.png")
print("="*55)