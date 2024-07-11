import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
            tf.keras.layers.Dense(512, activation="relu"),
            # 2nd
            tf.keras.layers.Dense(256, activation="relu"),
            # 3rd
            tf.keras.layers.Dense(64, activation="relu"),
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
    model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32,
    )
    # evaluate model

    # make predictions
