import tensorflow as tf
import numpy as np
from  keras.applications import ResNet50
from  keras.applications.resnet50 import preprocess_input as resnet_preprocess
from  keras.models import Sequential
from  keras.layers import Dense, GlobalAveragePooling2D, Dropout
from  keras.optimizers import Adam
from keras.models import save_model
data_dir = r"C:\Users\kumar\Desktop\Projects\PYTHON CODES\sahit\Dataset"
batch_size = 32
img_height = 224
img_width = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
classes=train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
model_choice = 'resnet50'  
def get_pretrained_model(choice):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    preprocess = resnet_preprocess
    base_model.trainable = False
    return base_model, preprocess
base_model, preprocess_fn = get_pretrained_model(model_choice)
def preprocess_ds(ds):
    return ds.map(lambda x, y: (preprocess_fn(x), y))
train_ds = preprocess_ds(train_ds)
val_ds = preprocess_ds(val_ds)
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(classes), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=5, validation_data=val_ds)
model.save(r"C:\Users\kumar\Desktop\Projects\PYTHON CODES\sahit\model.h5")
def load_and_preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)
    return img_array
def predict_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = classes[predicted_class]
    confidence = 100 * np.max(predictions[0])
    return predicted_label, confidence
# Test Prediction
path_input = input("Enter image path: ").strip('"')
label, conf = predict_image(path_input)
print(f"Predicted: {label} ({conf:.2f}% confidence)")