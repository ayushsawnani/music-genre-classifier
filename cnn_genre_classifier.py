import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plot

keras = tf.keras

DATASET_PATH = "data.json"


# load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert data into array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets


def prepare_datasets(test_size, validation_size):
    # load data
    x, y = load_data(DATASET_PATH)

    # create train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size
    )  # test set is 30% of all the dataset

    # create train validation split
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=validation_size
    )

    # 3d array -> (130 time bins, 13 MFCCs, 1 depth (amplitude))
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def predict(x, y, model):

    x = x[np.newaxis, ...]

    # probability [[0.1, 0.2, 0.4, ...]] for each genre
    prediction = model.predict(x)  # X ->(1, 130, 13, 1)

    # extract index with max vlaue

    index = np.argmax(prediction, axis=1)  # e.g. [4]

    print("Expected index: {}, Predicted index: {}".format(y, index))


def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # CNN with 3 convutional layers followed by max pooling layer
    # # of filters, kernel size

    model.add(
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(
        # kernel size
        keras.layers.MaxPool2D(
            (3, 3),
            strides=(2, 2),
            padding="same",
        )
    )
    model.add(keras.layers.BatchNormalization())

    model.add(
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(
        keras.layers.MaxPool2D(
            (3, 3),
            strides=(2, 2),
            padding="same",
        )
    )
    model.add(keras.layers.BatchNormalization())

    model.add(
        keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape)
    )
    model.add(
        keras.layers.MaxPool2D(
            (2, 2),
            strides=(2, 2),
            padding="same",
        )
    )
    model.add(keras.layers.BatchNormalization())

    # flatten out and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


if __name__ == "__main__":
    # create train, validation, and test sets
    # validation is the test set for the network to use to validate weights so that when against the test set it's never seen it befroe

    (
        inputs_train,
        inputs_validation,
        inputs_test,
        targets_train,
        targets_validation,
        targets_test,
    ) = prepare_datasets(0.25, 0.2)

    # build CNN

    # shape of 3d array
    inputs_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    model = build_model(inputs_shape)

    # compile network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # train the CNN
    model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_validation, targets_validation),
        batch_size=32,
        epochs=30,
    )

    # evaluate CNN on test set
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is {}".format(test_accuracy))

    # make predictions on a sample
    x = inputs_test[100]
    y = targets_test[100]
    predict(x, y, model)
