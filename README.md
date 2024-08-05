# OCTMNIST Classification

For this part, we will be working with a real-world dataset - OCTMNIST.

## OCTMNIST

The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. Each example is a 28x28 image, associated with a label from 4 classes.

### Getting the Data

MedMNIST is a collection of multiple datasets, for this assignment we will be working with one dataset from the collection – OCTMNIST.

- [MedMNIST Website](https://medmnist.com/)
- [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)
- [Direct Download](https://zenodo.org/record/6496656)

Install the MedMNIST package using:
```sh
pip install medmnist
```

**Data preprocessing**

- Preprocessing the OCTMNIST dataset is achieved by applying transformations to each image, such as resizing to 28x28 pixels, converting to a tensor, and normalizing with mean and standard deviation of 0.5. 
- On Normalizing the pixel values are scaled appropriately, improving the stability and performance of the neural network during training.
- 
**Splitting data to train, validation and testing data:**

- The transformed dataset is then split into training, validation, and testing sets to facilitate model evaluation and development.

**Model Architechture:**

The **OCTMNIST_CNN** model architecture comprises convolutional layers followed by ReLU activation functions and max-pooling, which reduce the spatial dimensions and introduce non-linearity to the model. This is followed by several fully connected layers, with ReLU activation functions applied after each layer to ensure non-linearity. 

**Regularization Techniques:**

- **Dropout layers** are incorporated between the fully connected layers to prevent overfitting by randomly setting a fraction of the input units to zero during training. - -- **Regularization techniques** like **L1 or L2 regularization** is applied to the model's parameters to penalize large weights and enhance generalization.
- **Early stopping** is  used to monitor the validation performance and halt training when the model's performance ceases to improve, thereby preventing overfitting.


 Model Summary:
 The following is a summary of the CNN model used for classifying OCTMNIST images:
```sh
  OCTMNIST_CNN(
(conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu1): ReLU()
(conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(relu2): ReLU()
(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(fc1): Linear(in_features=1568, out_features=256, bias=True)
(fc2): Linear(in_features=256, out_features=128, bias=True)
(fc3): Linear(in_features=128, out_features=64, bias=True)
(fc4): Linear(in_features=64, out_features=32, bias=True)
(fc5): Linear(in_features=32, out_features=4, bias=True)
(dropout): Dropout(p=0.1, inplace=False)
)
```


### Model Evaluation Results
  The evaluation results of the CNN model over 7 epochs are shown below:


## Training and Validation Accuracy
<img width="506" alt="accuracy_plot" src="https://github.com/user-attachments/assets/9993c3f8-e6fb-4019-825d-7b2fdf1be578">

## Training and Validation Loss
<img width="802" alt="loss graph-train-valid" src="https://github.com/user-attachments/assets/e1458e49-90ac-4ef6-843e-17feccf870e2">

## Confusion Matrix
<img width="408" alt="oct_confusion_matrix" src="https://github.com/user-attachments/assets/47a9d441-da4b-4427-bf76-46548c0593ad">
