# Object Localization with TensorFlow 
This project demonstrates how to build and train a **multi-output Convolutional Neural Network (CNN)** using TensorFlow/Keras to perform **object localization** and **classification simultaneously**.

Unlike full object detection systems, this implementation assumes **a single object per image** and predicts:
- The **class label** of the object  
- The **bounding box coordinates** of the object  

The model is trained entirely on **synthetically generated data** using emoji images.

## Project Overview
Object localization is a simplified version of object detection where:
- Each image contains exactly **one object**
- The model predicts:
  - A **class label** (classification task)
  - A **bounding box** (regression task)

This project builds a **dual-head CNN**:
- One output head for **classification**
- One output head for **bounding box regression**

## Key Features
- Synthetic dataset generation using emoji images
- Multi-output CNN using TensorFlow Keras Functional API
- Custom **IoU (Intersection over Union)** metric
- Custom **Keras callback** for visual model evaluation
- On-the-fly data generation using Python generators
- Visualization of predictions with bounding boxes

## Dataset & Data Generation
Instead of using a real dataset, this project generates data dynamically.

### Emoji Classes
The dataset consists of **9 emoji classes**, each mapped to a PNG file from the OpenMoji dataset.

### Synthetic Image Creation
Each training example is generated as follows:
- A blank **144×144 white image** is created
- A **72×72 emoji** is randomly selected
- The emoji is placed at a **random location**
- The model learns to:
  - Predict the **emoji class**
  - Predict the **top-left corner (row, col)** of the emoji

## Data Pipeline

### Example Generation
Each generated sample includes:
- `image`: Input image (144×144×3)
- `class_id`: Integer label (0–8)
- `bounding box`: (row, col)

### Data Generator
A custom generator yields batches in the format:

```python
(
  {'image': x_batch},
  {
    'class_out': y_batch,
    'box_out': bbox_batch
  }
)
```

Where:
- `x_batch`: Image tensor
- `y_batch`: One-hot encoded labels
- `bbox_batch`: Normalized bounding box coordinates

## Model Architecture
The model is built using the **Keras Functional API**.

### Input
- Shape: `(144, 144, 3)`

### Feature Extractor
A stack of convolutional blocks:
- Conv2D → ReLU
- BatchNormalization
- MaxPooling

The number of filters increases exponentially:
```
16 → 32 → 64 → 128 → 256
```

### Shared Backbone
After convolutional layers:
- Flatten layer converts features into a vector

### Output Heads

#### 1. Classification Head (`class_out`)
- Dense layers
- Softmax activation
- Output shape: `(9,)`

#### 2. Bounding Box Head (`box_out`)
- Dense layers
- Linear activation
- Output shape: `(2,)` → (row, col)

## Custom Metric: IoU (Intersection over Union)
A custom Keras metric is implemented to evaluate bounding box predictions.

### Purpose
Measures overlap between:
- Ground truth bounding box
- Predicted bounding box

### Implementation Highlights
- Maintains:
  - `total_iou`
  - `count`
- Computes IoU per batch
- Returns average IoU over time

## Model Compilation
The model is compiled with **multi-task learning objectives**:

```python
loss = {
  'class_out': 'categorical_crossentropy',
  'box_out': 'mse'
}
```

### Optimizer
- Adam (`learning_rate = 1e-3`)

### Metrics
- Classification: Accuracy
- Localization: Custom IoU

## Visualization & Debugging

### Bounding Box Plotting
A utility function overlays:
- Ground truth bounding boxes (green)
- Predicted bounding boxes (red)

## Custom Callback
A custom callback (`ShowTestImages`) is used to:
- Run inference after each epoch
- Display predictions on test samples

## Learning Rate Scheduling
A custom learning rate scheduler is implemented:
- Every 5 epochs:
  - Learning rate is reduced by a factor of **0.2**
- Lower bound: `3e-7`

## Training Pipeline
Training uses:
- `tf.data.Dataset.from_generator`
- Infinite data generation via Python generator
- Batch size: `16`

Each training step includes:
1. Generate synthetic batch
2. Forward pass through model
3. Compute:
   - Classification loss
   - Bounding box loss
4. Backpropagation

## Model Behavior
The model learns to:
- Accurately classify emojis
- Predict their spatial location

## Technologies Used
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Summary
This project demonstrates:
- How to design a **multi-output neural network**
- How to combine **classification + regression tasks**
- How to generate **synthetic training data**
- How to implement **custom metrics and callbacks**
- How to visualize model predictions effectively
