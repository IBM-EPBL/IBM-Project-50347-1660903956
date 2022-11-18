import pickle 
import cv2
from skimage import feature
from flask import Flask,request, render_template
import os.path

model = pickle.loads(open('C:/xampp/htdocs/Website/parkinson.pkl', "rb").read())
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def about():
    return render_template("C:/xampp/htdocs/Website/index.html")
@app.route("/about")
def home():
    return render_template("C:/xampp/htdocs/Website/index.html")
@app.route("/info")
def information():
    return render_template("C:/xampp/htdocs/Website/info.html")
@app.route("/predict")
def test():
    return render_template("C:/xampp/htdocs/Website/predict.html")
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f=request.files['file']
        basepath=os.path.dirname(file)
        filepath=os.path.join(basepath,"predict",f.filename)
        f.save(filepath)
        print("[INFO] loading model...")
        model = pickle.loads(open('C:/xampp/htdocs/Website/parkinson.pkl',"rb").read())
        image = cv2.imread(filepath)
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        # pre-process the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # quantify the image and make predictions based on the extracted features
        features = quantify_image(image)
        preds = model.predict([features])
        label = "Parkinsons" if preds[0] else "Healthy"
        color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        output_images.append(output)
        cv2.waitKey(0)
    return None
if __name__ == "__main__":
    app.run()