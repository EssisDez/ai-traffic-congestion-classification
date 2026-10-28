# ai-traffic-congestion-classification
Traffic congestion classification using CNN transfer learning with MobileNetV2 and Gradio.

# AI-Powered Traffic Congestion Classification Using CNN Transfer Learning

## 1. Project Overview

Traffic congestion is a common problem in urban areas. Heavy traffic can increase travel time, fuel consumption, air pollution, and road safety risks. Identifying traffic congestion levels from road images can help support traffic monitoring and transportation management.

This project develops an AI-powered traffic congestion classification system using **Convolutional Neural Networks (CNNs)** and transfer learning. The system uses the MobileNetV2 architecture to analyze traffic images and classify them into three categories:

- **Low Congestion:** Few vehicles are present, and traffic generally flows freely.
- **Moderate Congestion:** A moderate number of vehicles are present, and traffic may be slowing down.
- **High Congestion:** Many vehicles are closely packed, and traffic may be moving slowly or stopped.

The project also includes an interactive web interface built with Gradio, allowing users to upload a traffic image and receive a predicted congestion category with confidence scores.

## 2. Project Objectives

The main objectives of this project are:

1. To apply deep learning techniques to traffic image classification.
2. To implement CNN transfer learning using MobileNetV2.
3. To fine-tune a pretrained neural network for traffic congestion classification.
4. To classify road images into Low, Moderate, and High congestion categories.
5. To evaluate model performance using validation accuracy, a classification report, and a confusion matrix.
6. To develop an interactive interface that allows users to test the trained model.
7. To demonstrate a practical application of neural networks in intelligent transportation systems.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network construction and training |
| MobileNetV2 | Pretrained CNN architecture |
| NumPy | Numerical computations |
| Matplotlib | Visualization of training performance |
| Scikit-learn | Model evaluation |
| Gradio | Interactive web interface |
| Google Colab | Cloud-based development and GPU training |
| GitHub | Source code management and project sharing |

## 4. Deep Learning Methodology

### 4.1 Convolutional Neural Network (CNN)

A Convolutional Neural Network is a deep learning architecture commonly used for image recognition and classification. CNNs learn visual patterns such as edges, shapes, textures, and more complex image features.

In this project, the CNN analyzes traffic images to identify visual patterns associated with different congestion levels.

### 4.2 Transfer Learning

Transfer learning uses knowledge learned by a model on a large dataset to help solve a new task.

This project uses MobileNetV2, which is pretrained on ImageNet. Its learned visual features provide a starting point for classifying traffic images.

Transfer learning can reduce training time and the amount of data required compared with training an entire CNN from scratch.

### 4.3 MobileNetV2 Architecture

MobileNetV2 is a convolutional neural network designed to provide efficient image processing. It uses depthwise separable convolutions and inverted residual blocks to reduce computational requirements.

For this project, the original classification head is removed and replaced with a custom classification head that predicts three congestion categories.

### 4.4 Model Training

The training process consists of two main stages:

**Stage 1: Feature extraction**

- Load MobileNetV2 with pretrained ImageNet weights.
- Freeze the pretrained convolutional base.
- Add global average pooling, dropout, and a three-class softmax output layer.
- Train the new classification head on the traffic image dataset.

**Stage 2: Fine-tuning**

- Unfreeze selected later layers of MobileNetV2.
- Keep batch normalization layers frozen for stability.
- Train the model using a small learning rate.
- Monitor validation performance to reduce overfitting.

### 4.5 Image Preprocessing

The images are resized to 224 × 224 pixels to match the model input dimensions.

The pixel values are rescaled to the range expected by MobileNetV2. Data augmentation, including horizontal flipping, small rotations, and zooming, is used during training to help the model generalize to varied images.

### 4.6 Classification Output

The final layer uses the softmax activation function to generate a score for each class:

- Low congestion
- Moderate congestion
- High congestion

The interface displays these scores as confidence estimates. They should not be interpreted as guaranteed probabilities of real-world congestion.

## 5. System Workflow

The system follows this workflow:

1. Collect and label traffic images.
2. Organize images into three class folders.
3. Load and preprocess the dataset.
4. Build the MobileNetV2 transfer-learning model.
5. Train the classification head.
6. Fine-tune selected CNN layers.
7. Evaluate model performance.
8. Save the trained model.
9. Launch the Gradio interface.
10. Upload an image and display the predicted congestion class.

## 6. Dataset Structure

The notebook expects a dataset organized into three folders:

```text
traffic_dataset/
├── low/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── moderate/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── high/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The folder names represent the target classes. Each image must be assigned to the appropriate category based on its traffic conditions.

The dataset is divided into training and validation subsets. The training subset is used to optimize the model, while the validation subset is used to monitor its performance on images not used for training.

**Dataset source:** Add the name and link of the actual dataset used.

**Dataset size:** Add the total number of images and the number of images in each class after preparing the dataset.

**Dataset license:** Check and follow the original dataset's terms of use.

## 7. Installation and Requirements

The project was developed using Google Colab.

The required libraries can be installed using:

```bash
pip install tensorflow numpy matplotlib scikit-learn pandas pillow gradio
```

Alternatively, install the packages in a Colab notebook cell:

```python
!pip install -q gradio scikit-learn matplotlib
```

Google Colab already includes many common machine learning libraries.

## 8. How to Run the Project

### Step 1: Open the notebook

Open the project notebook in Google Colab.

**Notebook:** Add the path to your notebook in this repository.

### Step 2: Prepare the dataset

Prepare a ZIP file containing the `low`, `moderate`, and `high` folders. Upload the ZIP file when requested by the notebook.

### Step 3: Execute the notebook

Run the cells in order to:

- Import libraries.
- Load the dataset.
- Preprocess images.
- Build the MobileNetV2 model.
- Train and fine-tune the model.
- Evaluate its performance.
- Save the trained model.

### Step 4: Launch the interactive interface

Run the Gradio interface cell after training has completed.

Colab will provide a temporary web link if the interface launches successfully with sharing enabled.

### Step 5: Test the model

Upload an individual road image. The interface displays the predicted congestion category and scores for the three classes.

The interface requires the trained model to be available in the current runtime.

## 9. Interactive User Interface

The Gradio interface provides a simple way to interact with the trained CNN without manually writing prediction code.

The user can:

1. Open the interface link.
2. Upload a traffic image.
3. Submit the image for classification.
4. View the predicted congestion category.
5. Compare the model's scores for Low, Moderate, and High congestion.

The interface is intended as a prototype for demonstrating image classification using deep learning.

**Interface link:** Add your working Gradio link if you have a permanent hosting solution. Colab-generated sharing links are generally temporary.

## 10. Model Evaluation

The model is evaluated using a held-out validation dataset.

The notebook generates the following evaluation outputs:

### Validation accuracy

Validation accuracy measures the proportion of validation images classified correctly.

**Actual validation accuracy:** Add the value produced by your notebook.

### Validation loss

Validation loss measures the model's classification error according to the chosen loss function.

**Actual validation loss:** Add the value produced by your notebook.

### Classification report

The classification report provides precision, recall, and F1-score for each congestion class.

- **Precision:** The proportion of predictions for a class that are correct.
- **Recall:** The proportion of actual examples of a class that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.

Insert the actual classification report produced by your notebook.

### Confusion matrix

The confusion matrix compares actual classes with predicted classes. It helps identify which congestion categories are correctly classified and which are confused with one another.

Insert the confusion matrix generated by your notebook.

### Test image results

Record the actual results from your Gradio tests.

| Test | Actual class | Predicted class | Confidence |
|---|---|---|---|
| 1 | Add result | Add result | Add result |
| 2 | Add result | Add result | Add result |
| 3 | Add result | Add result | Add result |

These results must come from actual model predictions. Do not insert estimated or invented performance values.

## 11. Project Results

The completed system demonstrates how transfer learning can be applied to classify traffic images into three congestion categories.

The trained model's effectiveness must be assessed using the validation metrics and test predictions obtained during execution.

The final results should discuss:

- Validation accuracy and loss.
- Precision, recall, and F1-score for each class.
- Patterns observed in the confusion matrix.
- Performance on individual test images.
- Any differences in performance between congestion categories.

The quality and balance of the dataset can significantly affect these results.

## 12. Applications

Potential applications of traffic congestion image classification include:

- Traffic monitoring and analysis.
- Transportation research.
- Smart city prototypes.
- Road congestion awareness systems.
- Educational demonstrations of deep learning.
- Preliminary analysis of traffic camera images.

This project is a prototype and is not a complete real-time traffic management system.

## 13. Limitations

The system has several limitations:

1. **Dataset dependence:** Performance depends on the quality, quantity, and accuracy of the labeled images.
2. **Class ambiguity:** The distinction between Low, Moderate, and High congestion may be subjective unless clear labeling criteria are defined.
3. **Environmental variation:** Lighting, weather, camera angle, road type, and image quality may affect predictions.
4. **Generalization:** A model trained on one dataset may not perform equally well on different roads or locations.
5. **Image-based analysis:** The model classifies a single image rather than measuring traffic speed or congestion over time.
6. **No live traffic feed:** The current prototype does not automatically connect to live traffic cameras.
7. **Confidence scores:** Softmax scores are not necessarily calibrated probabilities of correctness.

## 14. Future Improvements

Possible improvements include:

- Expanding the dataset with more diverse traffic images.
- Improving class balance.
- Comparing MobileNetV2 with other CNN architectures.
- Applying additional evaluation methods and cross-validation.
- Testing the model on independent road images.
- Integrating a live traffic camera feed.
- Combining image classification with vehicle detection and tracking.
- Deploying the model through a persistent web application.
- Investigating model interpretability techniques such as Grad-CAM.

## 15. Ethical and Responsible Use

Traffic images should be collected and used in accordance with applicable privacy rules and dataset licenses.

The model's outputs should be treated as predictions rather than authoritative traffic measurements. Further validation would be required before using this system for operational transportation decisions.

## 16. Repository Structure

The repository may be organized as follows:

```text
ai-traffic-congestion-classification/
├── Traffic_Congestion_CNN_Colab.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    ├── training_accuracy.png
    └── confusion_matrix.png
```

The `results/` directory is optional. Add plots only after generating them from the actual model execution.

Do not commit private information, credentials, or large datasets without checking their licenses and repository storage requirements.

## 17. Conclusion

This project applies a CNN-based transfer-learning approach to traffic congestion image classification. MobileNetV2 provides pretrained visual features, while a custom classification head predicts Low, Moderate, and High congestion.

Fine-tuning allows selected pretrained layers to adapt to the target dataset. Model evaluation and an interactive Gradio interface make it possible to examine the classifier's performance and test it on individual images.

The project demonstrates a practical educational application of neural networks and transfer learning. Its real-world usefulness depends on rigorous evaluation, appropriate dataset labeling, and testing across diverse traffic conditions.

## 18. Author and Project Information

- **Project title:** AI-Powered Traffic Congestion Classification Using CNN Transfer Learning
- **Student name:** Add your name
- **Course:** Neural Networks and Deep Learning
- **Institution:** Add your institution
- **Academic year:** Add your academic year
- **GitHub repository:** Add your repository URL
- **Dataset source:** Add your dataset URL

---

*This project was developed for educational purposes as part of a Neural Networks and Deep Learning mini-project.*
