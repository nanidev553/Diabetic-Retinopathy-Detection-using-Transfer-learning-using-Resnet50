from werkzeug.routing import EndpointPrefix,RoutingException,BuildError
from werkzeug.security import check_password_hash,generate_password_hash
from flask.templating import render_template
from flask.app import Flask
from flask import request,url_for,redirect
import pymysql as sql
from keras.models import load_model
from  keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
import tensorflow as tf
model=load_model(filepath=r'C:\Users\kumar\Desktop\Projects\PYTHON CODES\sahit\model.h5')
data_dir = r"C:\Users\kumar\Desktop\Projects\PYTHON CODES\sahit\Dataset"
batch_size = 32
img_height = 224
img_width = 224
# Load dataset
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
filterwarnings('ignore')
app=Flask(__name__)
app.secret_key='drdkey'
@app.route('/')
@app.route('/Opti_Detect')
def home():
    return render_template('home.html')
@app.route('/detection')
def detection_page():
    return render_template('detection.html')
def load_and_preprocess_image(image_path):
    img_array = tf.keras.utils.img_to_array(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
def predict_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = classes[predicted_class]
    confidence = 100 * np.max(predictions[0])
    return predicted_label, confidence
@app.route('/image_analysis',methods=['POST','GET'])
def image_analysis():
    if request.method=='POST':
        img_input=request.files.get('image_input')
        img=Image.open(img_input)
        label,conf=predict_image(img)
        label_dict={
            'Mild':'Mild Diabetic Retinopathy',
            'Moderate':'Moderate Diabetic Retinopathy',
            'No_DR':'No Diabetic Retinopathy',
            'Proliferate_DR':'Proliferate Diabetic Retinopathy',
            'Severe':'Severe Diabetic Retinopathy'
        }
        result_dict={
            'diagnosis':label_dict[label],
            'confidence':conf,
            'severity_level':label,
            'model_version':'ResNet-50'
        }
        return render_template('results.html',result=result_dict)
    return render_template('detection.html')
app.run(debug=True,port=2023)