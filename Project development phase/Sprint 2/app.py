import sys
import os
import glob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from skimage.feature import hog
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import pickle
from flask import Flask, redirect, url_for, request, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open("parkinson.pkl","rb"))

def quantify_image(image):
    features = hog(image, orientations=9,
                           pixels_per_cell=(10, 10),
                           cells_per_block=(2, 2),
                           transform_sqrt=True,
                           block_norm="L1")
    return features

def predict_label(image):
    for i in image:
        features = quantify_image(image)
        preds = model.predict([features])
        label = "Parkinsons" if preds[0] else "Healthy"
        cv2.waitKey(0)
        return label

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        image=[]
        image.append(img_path)
        p = predict_label(img_path)
    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ == '__main__':
    app.run(debug=True)