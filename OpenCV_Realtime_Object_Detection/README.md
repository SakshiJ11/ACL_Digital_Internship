
# Pediatric Appendicitis Ultrasound Image Classification

This project aims to build a convolutional neural network (CNN) model that classifies ultrasound images into two categories: "Appendicitis" and "No Appendicitis." The dataset contains ultrasound images and corresponding medical records related to pediatric appendicitis.

## Project Overview

- **Dataset**: The project uses ultrasound images stored in BMP format and a CSV file with medical records that include information on Diagnosis, Management, and Severity.
- **Goal**: Build a deep learning model that can classify ultrasound images based on the diagnosis of appendicitis.

## Requirements

To run this project, ensure you have the following Python libraries installed:

- `tensorflow` (for building and training the neural network)
- `keras` (for high-level neural network API)
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `Pillow` (for image processing)
- `matplotlib` (for plotting results)
- `scikit-learn` (for train-test split)

You can install the necessary packages by running:

```bash
pip install tensorflow keras pandas numpy Pillow matplotlib scikit-learn
```

## Project Structure

The project consists of the following components:

- **`data.csv`**: CSV file containing the dataset with medical records (Diagnosis, Management, Severity).
- **`US_Pictures/`**: Folder containing the ultrasound images in BMP format.
- **`cleaned_data.csv`**: Output CSV file after cleaning and preprocessing the data.
- **`model.py`**: Python script to load data, preprocess images, build, train, and evaluate the CNN model.
- **`README.md`**: Documentation for the project.

## Steps

### 1. Data Loading and Preprocessing

- The dataset is read from a CSV file (`data.csv`), and missing values in the target columns (Diagnosis, Management, Severity) are handled.
- Each ultrasound image is matched to the corresponding subject ID in the dataset.
- Images are preprocessed: resized to 224x224 pixels and normalized to a range of [0, 1].
- The images are labeled based on the Diagnosis column (1 for appendicitis, 0 for non-appendicitis).

### 2. Model Building

- A CNN is created with 3 convolutional layers, followed by max-pooling layers.
- The model is compiled using the Adam optimizer and binary cross-entropy loss function for binary classification.

### 3. Model Training

- The model is trained on the preprocessed data for 10 epochs.
- Training and validation accuracy and loss are plotted to visualize the model's performance.

### 4. Model Evaluation

- The model is evaluated on the test data, and the accuracy is printed.

### 5. Prediction for New Image

- You can use the trained model to predict whether a new ultrasound image indicates appendicitis.

## How to Use

1. **Load and Preprocess Data**: The script automatically loads and preprocesses the images and data, matching each image with its corresponding labels.
2. **Train the Model**: The model is trained on the images with the target labels.
3. **Evaluate the Model**: After training, the model is evaluated on the test set to print accuracy and plot training/validation metrics.
4. **Predict New Images**: To predict the class of a new image, replace the `image_path` variable with the path to your image file and run the script.

### Example:

```python
# For a new image
image_path = "/path/to/your/image.bmp"
new_image = Image.open(image_path).convert("RGB").resize((224, 224))
new_image = np.array(new_image) / 255.0
new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(new_image)
if prediction > 0.5:
    print("Appendicitis")
else:
    print("No Appendicitis")
```

## Model Summary

- **Architecture**:
    - 3 Convolutional Layers with ReLU activation and MaxPooling
    - Flatten layer to convert 2D features to 1D
    - Dense layer with 128 units and ReLU activation
    - Output layer with sigmoid activation for binary classification




