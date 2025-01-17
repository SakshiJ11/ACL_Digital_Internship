
# Age and Gender Detection Using PyTorch

This repository provides an implementation of a multi-task deep learning model that predicts a person's age (as a range) and gender from facial images using PyTorch. The dataset used for training and evaluation is UTKFace.

## Features

- **Gender Detection**: Binary classification for male and female gender.
- **Age Range Prediction**: Classifies the input into one of the predefined age ranges.
- **Age Estimation**: Predicts the exact age as a regression task.
- Preprocessing pipeline using PyTorch's `torchvision`.
- Custom dataset class for UTKFace.
- Training, validation, and evaluation functions.
- Scheduler to reduce learning rates dynamically based on validation performance.

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- NumPy
- OpenCV
- PIL (Pillow)
- scikit-learn
- tqdm

Install the required packages using pip:

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn tqdm
```

## Dataset

The model uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/) for training and evaluation. Each image filename follows the format:

```
<age>_<gender>_<race>_<date>.jpg
```

- `age`: Age of the person in years.
- `gender`: 0 for male, 1 for female.
- `race`: Not used in this implementation.

Place the dataset in the `Dataset/UTKFace/` directory, or update the `root_dir` variable in the script.

## Model Architectures

### 1. **Gender Model**
- Based on VGG-Face.
- Predicts gender (male/female) using binary classification.

### 2. **Age Range Model**
- A convolutional neural network (CNN) with three convolutional blocks.
- Predicts one of 9 age ranges:
  - `[0-4, 5-9, 10-15, 16-25, 26-35, 36-45, 46-60, 61-75, 76+]`

### 3. **Age Estimation Model**
- Uses embeddings of predicted age ranges.
- Predicts exact age using regression.

## Training

The script trains the three models jointly using a custom training function.

1. Load the dataset using the `UTKFaceDataset` class.
2. Split the data into training and validation sets.
3. Train using the `train()` function:

```python
train(
    gender_model=gender_model,
    age_range_model=age_range_model,
    age_estimation_model=age_estimation_model,
    train_data=trainloader,
    device='cuda',
    weights=None,
    num_epochs=50,
    validation_split=0.2,
    save_last_weights='AgeGenderWeights.pt'
)
```

### Loss Functions

- Gender Detection: Binary Cross-Entropy Loss.
- Age Range Prediction: Cross-Entropy Loss.
- Age Estimation: L1 Loss.

### Optimizers

- Adam optimizer with initial learning rates:
  - Gender Model: `1e-4`
  - Age Range Model: `5e-3`
  - Age Estimation Model: `1e-3`

### Learning Rate Schedulers

- ReduceLROnPlateau to dynamically adjust learning rates based on validation loss.

## Evaluation

Evaluate the trained models on the validation set and generate a confusion matrix for age range prediction:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

## Results

After training, the model outputs:

- Loss and accuracy metrics for gender and age range prediction.
- Regression loss for age estimation.
- Confusion matrix for age range classification.

## How to Use

1. Clone this repository:

```bash
git clone https://github.com/yourusername/age-gender-detection.git
cd age-gender-detection
```

2. Place the UTKFace dataset in the `Dataset/UTKFace/` directory.

3. Run the script:

```bash
python main.py
```

4. Check the saved weights (`AgeGenderWeights.pt`) for inference or further fine-tuning.

## To Do

- Add support for real-time inference using webcam or video input.
- Improve the performance of age estimation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- PyTorch community for tutorials and resources.

