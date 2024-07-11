# Music Genre Classifier

## Overview

This repository contains a music genre classifier built using TensorFlow, Keras, and scikit-learn. The goal of this project is to develop a neural network model capable of predicting the genre of 30-second music samples. The dataset consists of multiple genres, and the model is trained and tested using a train/test split approach.

## Features

- **Neural Network Model**: Built with TensorFlow and Keras, the model is designed to classify music samples into their respective genres.
- **Data Processing**: Utilizes scikit-learn's train/test split to efficiently partition the dataset for training and testing.
- **30-Second Samples**: The model is specifically trained on 30-second clips of music, ensuring consistent input length.
- **Multi-Genre Classification**: Capable of distinguishing between multiple music genres based on the training data.

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

## Progress

Using an input layer, 3 hidden layers, and an output layer, I have finished training the dataset.
<img width="621" alt="Training" src="https://github.com/ayushsawnani/music-genre-classifier/assets/81490699/afe86d08-f14e-403d-b665-93d9afcaae91">

The neural network has finished training, however I have encountered a problem known as **overfitting**, or "when an algorithm fits too closely or even exactly to its training data, resulting in a model that can't make accurate predictions or conclusions from any data other than the training data" (IBM). The greatest indicator of this is the "accuracy" being above 98%, while the validation accuracy, otherwise known as the accuracy on the testing data, is just under 59%.
<img width="826" alt="Overfitting" src="https://github.com/ayushsawnani/music-genre-classifier/assets/81490699/c5e4e4a2-66d7-428e-a1c1-722b45fab82d">

A more visual representation of overfitting is shown here:

## Future

I would like to implement a CNN (Convolutional Neural Network) and an RNN-LSTM (Recurrent Neural Network - Long Term Short Memory Network) to predict data more accurately.

## Acknowledgements

A special thank you to Valerio Velardo and his incredible YouTube course, [Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf). His tutorials and insights have been instrumental in the development of this music genre classifier project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
