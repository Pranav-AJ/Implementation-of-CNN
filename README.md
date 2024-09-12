# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), totaling 60,000 training images and 10,000 test images.

## Neural Network Model

![image](https://github.com/user-attachments/assets/b562945c-2825-4f20-aa8a-f8285d4e17be)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and Load the data set.

### STEP 2:
Reshape and normalize the data.

### STEP 3:
In the EarlyStoppingCallback change define the on_epoch_end funtion and define the necessary condition for accuracy

### STEP 4:
Train the model

## PROGRAM

### Name: A.J.PRANAV
### Register Number: 212222230107
```
import numpy as np
import tensorflow as tf

# Provide path to get the full path
data_path ="/content/mnist.npz"

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """
    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(60000,28,28,1)
    
    # Normalize pixel values
    images = images/255
    
    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name:            RegisterNumber:          \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


# EarlyStoppingCallback
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self,epoch,logs=None):
        
        # Check if the accuracy is greater or equal to 0.995
        if logs['accuracy'] >= 0.995:
                            
            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\nReached 99.5% accuracy so cancelling training!\n")
            print('Name:A.J.PRANAV            Register Number:212222230107   \n')

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """


    # Define the model
    model = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(28,28,1)),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu') ,
        tf.keras.layers.Dense(10,activation="softmax")
    ]) 

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
          
    return model

model = convolutional_model()

training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
```

## OUTPUT

### Reshape and Normalize output

![image](https://github.com/user-attachments/assets/f16e3b5a-572f-46ed-bc0f-cb1a5633c281)


### Training the model output

![image](https://github.com/user-attachments/assets/fc09d932-3f30-45b8-83be-ce8dd60ad309)



## RESULT

Hence  a convolutional deep neural network for digit classification was successfully developed.
