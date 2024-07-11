import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plot

DATASET_PATH = "data.json"


# load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert data into array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def generate_dataset(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size
    )  # test set is 30% of all the dataset
    return x_train, x_test, y_train, y_test


def plot_history(history):
    figure, axis = plot.subplots(2)

    # create accuracy subplot
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    # create error subplot
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    plot.show()


if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)

    # split data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = generate_dataset(
        inputs, targets, 0.3
    )

    # build the neural network
    model = tf.keras.Sequential(
        [
            # 2 dimensional flattening to one dimension
            tf.keras.layers.Flatten(
                input_shape=(inputs.shape[1], inputs.shape[2]),
            ),
            # 1st hidden layer
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.3),
            # 2nd
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.3),
            # 3rd
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.3),
            # output layer (softmax just sums the value and normalizes it to 1 (highest value))
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )  # keras makes model easy to build, sequential network

    # compile network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # train network
    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32,
    )

    # plot accuracy and error over the epochs

    plot_history(history)

    # evaluate model

    # make predictions
