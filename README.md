# Music Genre Classifier

This project is a Music Genre Classifier that utilizes TensorFlow, Keras, and scikit-learn to build and train a neural network model. The model is designed to predict the genre of a 30-second music sample using Mel-Frequency Cepstral Coefficients (MFCC) as input features.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier
```

## Usage

1. **Prepare Data**: Ensure you have a dataset of 30-second music samples, labeled with their respective genres. I downloaded the GTZAN Dataset for Music Genre Classification (found at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)
2. **Preprocess Data**: Use the dataset and process the data into a .json file
   ```bash
   python preprocess.py
   ```
3. **Train the Model**: Use the .json file to train the neural network
   ```bash
   python genre_classifier.py
   ```

## Dependencies

- TensorFlow
- Keras
- scikit-learn
- NumPy
- Json

## Code Overview

### Loading Data

The `load_data` function loads the dataset from the JSON file and converts it into numpy arrays for inputs (MFCC features) and targets (genre labels).

```
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert data into array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets
```

### Splitting Data

The `generate_dataset` function splits the data into training and testing sets using an 70/30 split.

```
def generate_dataset(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test
```

### Building the Neural Network

The neural network is built using Keras' Sequential API. It consists of several dense layers with ReLU activation and dropout for regularization.

```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
    tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

### Compiling and Training the Model

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric. It is then trained for 50 epochs with a batch size of 32.

```
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32)
```

### Plotting Training History

The `plot_history` function plots the accuracy and error over the epochs for both training and validation sets.

```
def plot_history(history):
    figure, axis = plot.subplots(2)
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    plot.show()
```

## Progress

Using an input layer, 3 hidden layers, and an output layer, I have finished training the dataset.
<img width="621" alt="Training" src="https://github.com/ayushsawnani/music-genre-classifier/assets/81490699/afe86d08-f14e-403d-b665-93d9afcaae91">

The neural network has finished training, however I have encountered a problem known as **overfitting**, or "when an algorithm fits too closely or even exactly to its training data, resulting in a model that can't make accurate predictions or conclusions from any data other than the training data" (IBM). The greatest indicator of this is the "accuracy" being above 98%, while the validation accuracy, otherwise known as the accuracy on the testing data, is just under 59%.

<img width="826" alt="Overfitting" src="https://github.com/ayushsawnani/music-genre-classifier/assets/81490699/c5e4e4a2-66d7-428e-a1c1-722b45fab82d">

A more visual representation of overfitting is shown here:

<img width="711" alt="Screenshot 2024-07-11 at 4 10 01 PM" src="https://github.com/user-attachments/assets/d478795c-642d-4e83-9647-3a4a1ea39372">

The top graph is the accuracy of the training data (blue) and the testing data (orange). As you can see, the accuracies of both differ greatly with each epoch. The bottom graph visualizes the same problem, but through the model's errors.

### Solving Overfitting

To address overfitting in the neural network model, I implemented dropout and L2 regularization techniques:

- Dropout: Added dropout layers to the model to randomly drop a fraction of the neurons during training, which helps prevent the model from becoming too dependent on specific neurons.
- L2 Regularization: Applied L2 regularization to the weights of the model to penalize large weights and encourage the model to maintain smaller weights, reducing the risk of overfitting.

These techniques helped improve the model's generalization to unseen data.

<img width="711" alt="Screenshot 2024-07-11 at 4 12 12 PM" src="https://github.com/user-attachments/assets/ea32b928-983e-406b-80e2-29ab046f1bc5">

The new graphs show a smaller difference from the training set to the testing set.

## Future

I would like to implement a CNN (Convolutional Neural Network) and an RNN-LSTM (Recurrent Neural Network - Long Term Short Memory Network) to predict data more accurately.

## Acknowledgements

A special thank you to Valerio Velardo and his incredible YouTube course, [Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf). His tutorials and insights have been instrumental in the development of this music genre classifier project.

## License

This project is licensed under the MIT License.
