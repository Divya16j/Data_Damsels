# Data_Damsels 
# Screening of eye disease using retinal images 
  Our project aims to develop an efficient screening system using transfer learning algorithms to detect and classify eye diseases like diabetic retinopathy, macular degeneration, and glaucoma. The current standard method of manual examination by ophthalmologists is time-consuming, costly, and causes an  errors. Through transfer learning, we plan to leverage pre-trained deep learning models on large-scale datasets to create a robust and accurate model for automated eye disease detection.
# Transfer Learning technique:
   we utilize a pre-trained CNN model as the foundation and fine-tune it for our specific  retinal eye disease detection task.
# Importing Libraries:
  The code starts by importing necessary libraries, such as pandas, numpy, and TensorFlow/Keras libraries for building and training the deep learning model.
# Loading Data and Preprocessing:
  The code reads the CSV file containing image labels and paths to the training images.It preprocesses the data by loading images, converting them to arrays, and splitting the dataset into training and validation sets.
# Data Augmentation:
  Data augmentation is applied using the ImageDataGenerator from Keras. It performs random transformations on the training data, such as rotation, width and height shifts, shear, zoom, and horizontal flip. This helps in increasing the variety of training samples and generalization of the model.
# Used Transfer Learning with VGG16:
# The process of model compilation and training has been successfully finished, and the resulting model has been saved with a ".h5" extension.
