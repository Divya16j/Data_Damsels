import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder

# Set paths and parameters
csv_file_path = "C:/Users/appu/Desktop/application building/Test_Set/Test_Set/Processed_RFMiD_Testing_Labels.csv"
image_folder_path = "C:/Users/appu/Desktop/application building/Test_Set/Test_Set/Test"
image_size = (128, 128)  # Adjust the size according to your requirement

# Load the CSV file containing image labels
df = pd.read_csv(csv_file_path)

# Preprocess the data
X_test = []  # List to store test image arrays
image_names = []  # List to store image names for output
for index, row in df.iterrows():
    image_path = os.path.join(image_folder_path, row['Image_Name'])
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    image_names.append(row['Image_Name'])

X_test = np.array(X_test)

# Load the saved model
model = load_model("final_model1.h5")

# Load label encoder classes
le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes1.npy", allow_pickle=True)

# Get input image paths from the user
num_images = int(input("Enter the number of input images: "))
input_image_paths = []
for i in range(num_images):
    input_image_path = input(f"Enter the path of input image {i+1}: ")
    input_image_paths.append(input_image_path)
# Create a dictionary to map disease abbreviations to full names
disease_mapping = {
    "ARMD" : "Age-related macular degeneration",
    "MH" :"Media haze",
    "DN" : "Drusen",
    "MYA" : "Myopia",
    "BRVO" : "Branch retinal vein occlusion",
    "TSLN ":"Tessellation",
    "ERM ": "Epiretinal membrane",
    "LS" : "Laser scars",
    "MS" : "Macular scars",
    "CSR" : "Central serous retinopathy",
    "ODC" :"Optic disc cupping",
    "CRVO ": "Central retinal vein occlusion",
    "TV" : "Tortuous vessels",
    "AH" : "Asteroid hyalosis",
    "ODP" :"Optic disc pallor",
    "ODE" : "Optic disc edema",
    "ST" :"Optociliary shunt",
    "AION ":"Anterior ischemic optic neuropathy",
    "PT" :"Parafoveal telangiectasia",
    "RT" : "Retinal traction",
    "RS" : "Retinitis",
    "CRS" : "Chorioretinitis",
    "EDN" : "Exudation",
    "RPEC" : "Retinal pigment epithelium changes",
    "MHL" : "Macular hole",
    "RP" : "Retinitis pigmentosa",
    "CWS" : "Cotton-wool spots",
    "CB" : "Coloboma",
    "ODPM" : "Optic disc pit maculopathy",
    "PRH" : "Myelinated nerve fibers",
    "HR" : "Hemorrhagic retinopathy",
    "CRAO" : "Central retinal artery occlusion",
    "TD" :"Tilted disc",
    "CME" :"Cystoid macular edema",
    "PTCR" : "Post-traumatic choroidal rupture",
    "CF ": "Choroidal folds",
    "VH" : "Vitreous hemorrhage",
    "MCA" : "Macroaneurysm",
    "VS ": "Vasculitis",
    "BRAO ": "Branch retinal artery occlusion",
    "PLQ ": "Plaque",
    "HPED" : "Hemorrhagic pigment epithelial detachment",
    "CL" : "Collateral"
    # Add more mappings for other diseases if needed
}
# Process each user-provided image, make predictions, and display results
for input_image_path in input_image_paths:
    # Load and preprocess the user-provided image
    user_img = Image.open(input_image_path).convert("RGB")
    user_img = user_img.resize(image_size)
    user_img_array = np.array(user_img) / 255.0

    # Reshape the user image to match the model input shape
    user_img_array = np.expand_dims(user_img_array, axis=0)

    # Make prediction on the user-provided image
    predicted_label_idx = np.argmax(model.predict(user_img_array), axis=-1)[0]
    predicted_disease = le.inverse_transform([predicted_label_idx])[0]

    # Display the output image along with the predicted disease name
    plt.imshow(user_img)
    plt.title(f"Predicted Disease: {predicted_disease}")
    plt.axis('off')
    plt.show()