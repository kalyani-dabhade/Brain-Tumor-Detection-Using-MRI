"""Optional model training starter.
Expected folders:
 dataset/Training/<class_name>/*.jpg
 dataset/Testing/<class_name>/*.jpg
Run only after installing a compatible TensorFlow version.
"""
import os, json
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET = os.path.join(os.path.dirname(__file__), 'dataset')
TRAIN_DIR = os.path.join(DATASET, 'Training')
TEST_DIR = os.path.join(DATASET, 'Testing')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
    raise SystemExit('Create dataset/Training and dataset/Testing folders, each with one subfolder per class.')
train = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
class_names = train.class_names
normalization = layers.Rescaling(1./255)
base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
base.trainable = False
model = models.Sequential([layers.Input(shape=IMG_SIZE+(3,)), normalization, base,
    layers.GlobalAveragePooling2D(), layers.Dropout(0.25), layers.Dense(len(class_names), activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=test, epochs=EPOCHS)
model.save(os.path.join(os.path.dirname(__file__), 'models', 'brain_tumor_model.keras'))
with open(os.path.join(os.path.dirname(__file__), 'models', 'class_names.json'), 'w', encoding='utf-8') as f:
    json.dump(class_names, f)
print('Saved model and class names. Check validation performance before any use.')
