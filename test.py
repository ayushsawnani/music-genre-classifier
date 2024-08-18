import json
import os
import numpy as np
from scipy import stats as st
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plot
import lstm, preprocess


keras = tf.keras

DATASET_PATH = "test_song/"
JSON_PATH = "test.json"

GENRES = [
    "pop",
    "metal",
    "disco",
    "blues",
    "reggae",
    "classical",
    "rock",
    "hiphop",
    "country",
    "jazz",
]


# load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert data into array
    inputs = np.array(data["mfcc"])
    os.remove("test.json")
    return inputs


def predict(x, model):

    array = []

    for input in x:

        input = input[np.newaxis, ...]

        # probability [[0.1, 0.2, 0.4, ...]] for each genre
        prediction = model.predict(input)  # X ->(1, 130, 13, 1)

        # extract index with max vlaue

        index = np.argmax(prediction, axis=1)  # e.g. [4]

        print("Predicted index: {}".format(index[0]))
        array.append(index[0])
    return "The model predicts that this is a {} song.".format(
        GENRES[st.mode(array).mode]
    )


def run():
    # preprocess song
    preprocess.save_mfcc_single(DATASET_PATH, JSON_PATH)

    # create model
    model = lstm.get_model()

    # compile network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # checkpoint path
    checkpoint_path = "training_1/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )

    model.load_weights(checkpoint_path)

    # # make prediction off of song

    inputs = load_data("test.json")
    return predict(inputs, model)
