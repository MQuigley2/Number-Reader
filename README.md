# Number-Reader
## Purpose

The aim of this project is to create a Neural Network which can accurately recognize handwritten digits, and to integrate it into a graphical application which allows users to draw draw their own digits to be evaluated.

## Libraries Used

The pandas library is used to load and transform training and validation set data.

NumPy arrays are used to feed data into the Neural Network.

TensorFlow's keras API is used to contruct the Neural Network.

Tkinter is used to create the Graphical User Interface.

scikit-image's io module is used to transform Tkinter canvas into NumPy array.

## Method

The model was trained on the publicly available MNIST dataset. Data was acquired from [kaggle](https://www.kaggle.com/c/digit-recognizer/data). The data set was split into a training and validation set with 3600 of 42000 data point being included in the validation set.

The model consists of two Convolutional layers, followed by pooling and dropout layers, which feed into a second set of two Convolutional layers followed by pooling and dropout layers. The data is then flattened and passes through a single fully connected layer and a final dropout layer before it reaches the output layer.  Initial tests used a low dropout rate of 0.1, but this failed to prevent overfitting. The dropout rate was increased to 0.3, which solved overfitting issues. Initial tests also included a second fully connected layer and used more features, but this did not seem to improve performance and only slowed down training.

Canvas is converted into array of pixel brightnesses which is clipped to remove whitespace, then inverted and normalized. Image array is then resized to 28x28 to be fed into Neural Network.

## Results
Neural Network achieved accuracy of 99.3% on validation set after training for 21 epochs. When evaluating digits drawn through GUI, it seems to be less accurate, but still works fairly well for most digits. In particular numbers which can be written in multiple ways seem to be difficult. 7 is often misclassified except when drawn with an extra horizontal line in the middle. 4 is also often misclassified depending on how it is drawn. The discrepancy in accuracy between test set and application indicates more work is needed on the image processing pipeline.


