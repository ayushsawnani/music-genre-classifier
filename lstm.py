import json
import os
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

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(
        keras.layers.LSTM(
            64,
            input_shape=input_shape,
            return_sequences=True,
        )
    )
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def get_model():
    inputs_shape = (130, 13)  # 130, 13
    model = build_model(inputs_shape)
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

    # # build RNN

    model = get_model()

    # compile network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # # save trained data
    checkpoint_path = "training_1/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    # # train the RNN
    # model.fit(
    #     inputs_train,
    #     targets_train,
    #     validation_data=(inputs_validation, targets_validation),
    #     batch_size=32,
    #     epochs=30,
    #     callbacks=[cp_callback],
    # )

    # # evaluate RNN on test set
    # test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    # print("Accuracy on test set is {}".format(test_accuracy))

    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    test2_error, test2_accuracy = model.evaluate(inputs_test, targets_test, verbose=2)
    print("Accuracy on restored test set is {}".format(test2_accuracy))

    # so first we want to input a sample
    # preprocess the sample
    # use the predict function to predict the genre
    # find out which index they predicted and return value

    # make predictions on a sample
    # x = inputs_test[100]
    # y = targets_test[100]
    # predict(x, y, model)
