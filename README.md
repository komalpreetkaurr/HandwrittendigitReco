***Handwritten Digit Recognition (MNIST)***

This project implements a Handwritten Digit Recognition system using the MNIST dataset. The model is built using TensorFlow/Keras and utilizes a Convolutional Neural Network (CNN) for accurate classification of digits (0-9).

📌 ***Features***

Uses MNIST dataset for training and testing.

CNN-based model for high accuracy.

Preprocessing of images (normalization & reshaping).

Model training & evaluation included.

Can be extended for real-world digit recognition tasks.

🛠️ ***Installation***

Ensure you have Python 3.7+ and the following dependencies installed:
```
pip install tensorflow numpy matplotlib
```

📊 ***Model Architecture***

The CNN model consists of:

2 Convolutional Layers (ReLU activation, MaxPooling)

Flatten Layer

Dense Layers for classification

model = tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

📌 ***Dataset***

We use the MNIST dataset, which contains 60,000 training and 10,000 testing images of handwritten digits. The dataset is available through Keras:

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

📈 Training & Performance

After 10-15 epochs, the model achieves ~98% accuracy on the test set.

Training Accuracy
```
~99%
```
Test Accuracy
```
~98%
```
⭐ ***Contributing***
This project was developed through collaborative efforts. Special thanks to [**Vinay10100**] for their meticulous code contributions. You’re welcome to contribute by refining the model, expanding datasets, or enhancing the documentation. Fork the repository and submit a pull request!

📞 **Contact**

For any queries, contact [KomalpreetKaur] at [kpreetk.879@gmail.com].
