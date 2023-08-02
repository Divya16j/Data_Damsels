import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder classes
model = load_model("final_model.h5")
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)

# Function to preprocess the image and make predictions
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predicted_label_idx = np.argmax(model.predict(img_array), axis=-1)[0]
    predicted_disease = le.inverse_transform([predicted_label_idx])[0]
    additional_info = "Retinal diseases are a group of eye conditions that affect the retina, which is the light-sensitive tissue located at the back of the eye.Treatment options for retinal diseases vary depending on the specific condition and its severity. Some common treatments include medications, laser therapy, photodynamic therapy, intravitreal injections, and surgery. "
    return img, predicted_disease, additional_info

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "image" in request.files:
        image = request.files["image"]
        if image.filename == "":
            return redirect(request.url)
        image_path = os.path.join("static/images", "user_image.png")
        image.save(image_path)
        return redirect(url_for("result"))
    return render_template("index.html")

@app.route("/result")
def result():
    img, predicted_disease, additional_info = process_image("static/images/user_image.png")
    return render_template("result.html", result_image="static/images/user_image.png",
                           predicted_disease=predicted_disease, additional_info=additional_info)

if __name__ == "__main__":
    app.run(debug=True)
