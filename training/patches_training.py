import os
import pickle
import argparse

import numpy as np

from sklearn.utils import class_weight

import tensorflow as tf

print("[INFO] Successfully imported Tensorflow version: {}".format(tf.__version__))
print("[DEBUG] Checking for GPU... {}".format(tf.config.list_physical_devices("GPU")))

# Set GPU memory to growth
gpus = tf.config.list_logical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="Patch neural network training for breast cancer prediction.")
parser.add_argument("dataset", help="The patch dataset we want to train on.")

args = parser.parse_args()

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
    
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join("../datasets/", args.dataset, "train/"),
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode="categorical"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join("../datasets/", args.dataset, "validation/"),
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode="categorical"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join("../datasets/", args.dataset, "test/"),
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

# Calculate class weights
y = np.argmax(np.concatenate([y for x, y in train_dataset], axis=0), axis=1)
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Input preprocessing
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Create the base model from pre-trained weights
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
    classes=5
)
base_model.trainable = False

# Create the model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
METRICS = [
    'accuracy',
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = METRICS
)

# Train the model
print("[INFO] Starting 3 epochs training")
history = model.fit(
    train_dataset,
    epochs=3,
    validation_data=validation_dataset,
    class_weight=class_weights
)

# Save the history
with open("../models/histories/" + args.dataset + "/" + "1.pkl", 'wb') as file:
    pickle.dump(history.history, file)

# Fine tuning
for layer in base_model.layers[46:]:
    # Unfreeze the top layers
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = METRICS
)

# Continue training
print("[INFO] Starting 10 epochs training")
history = model.fit(
    train_dataset,
    epochs=13,
    initial_epoch=3,
    validation_data=validation_dataset,
    class_weight=class_weights)

# Save the history
with open("../models/histories/" + args.dataset + "/" + "2.pkl", 'wb') as file:
    pickle.dump(history.history, file)

# Finish Training
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10e-5),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = METRICS
)

# Early stopping and weight saving
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = "../models/weights/patches/" + args.dataset + "/" + ".h5",
    monitor = "val_loss",
    save_best_only = True,
    verbose = 1
)

early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 10,
    verbose = 1,
)

# Continue training
print("[INFO] Starting 37 epochs training")
history = model.fit(
    train_dataset,
    epochs=50,
    initial_epoch=13,
    validation_data=validation_dataset,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopper]
)

# Save the history
with open("../models/histories/" + args.dataset + "/" + "3.pkl", 'wb') as file:
    pickle.dump(history.history, file)
