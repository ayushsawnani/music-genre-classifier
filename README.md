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
- librosa
- NumPy
- Json
- Flask

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

#### MLP (Multi-Layer Perception)
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

#### CNN (Convolutional Neural Network)
A convolutional neural network (CNN) consists of multiple convolutional and max-pooling layers, batch normalization, followed by dense and dropout layers. It uses a kernel, or a filter, that is applied to the spectrogram to detect certain features. Pooling is used to control overfitting and computer load by downsampling the spectrogram. Max pooling retains the most significant feature while reducing the size of the map.

```

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

```


### Compiling and Training the Model

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric. It is then trained for 50 epochs with a batch size of 32.

```
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32)
```

### Plotting Training History (MLP ONLY)

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

### Upgrading to a Convolutional Neural Network (CNN)

To improve the performance of the music genre classifier, I implemented a convolutional neural network (CNN) instead of a traditional multi-layer perceptron (MLP). The CNN was chosen because it excels at capturing spatial and temporal patterns, making it more suitable for processing audio spectrograms derived from 30-second music samples. 

<img width="883" alt="Screenshot 2024-07-14 at 3 44 10 PM" src="https://github.com/user-attachments/assets/a9dd8650-8d57-4440-af8e-297097bfc68c">
<img width="883" alt="Screenshot 2024-07-14 at 3 38 50 PM" src="https://github.com/user-attachments/assets/8e64b492-27b9-4b29-826e-b5f6087fafc2">

Network | Accuracy
:---: | :---:
MLP | 18.66%
CNN | 71.60%

This approach significantly improved the model's ability to recognize and classify complex audio patterns, resulting in higher accuracy for music genre prediction.

### Implementing an RNN-LSTM
To further enhance the model's performance, I implemented a Recurrent Neural Network with Long Short-Term Memory (RNN-LSTM). This architecture is particularly well-suited for sequential data, such as audio, as it can capture temporal dependencies more effectively than other models. The LSTM layers help the model remember long-term patterns, which is crucial for accurately predicting music genres over time.

The RNN-LSTM model architecture includes:

LSTM Layers: Stacked LSTM layers with dropout to handle the sequential nature of the audio data.
Dense Layers: Fully connected layers to map the learned features to the genre labels.

```
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

```

This RNN-LSTM architecture should allow the model to achieve even better accuracy on music genre classification. However, it's crucial to monitor the training process to prevent overfitting and ensure that the model generalizes well to unseen data.

The training results and accuracy metrics will be added after completing the experiments.


## Deploying the Music Genre Classifier as a Web App with Flask

To make the Music Genre Classifier accessible to users, I created a web application using Python's Flask framework, along with HTML and CSS for the frontend. This web app allows users to upload their own audio files, and the trained model will predict the genre of the uploaded music.

### Building the Web App

The web app consists of two main components:

Upload Form (HTML): A simple HTML form for uploading audio files.
Prediction Logic (Flask): Python code to handle the uploaded file, preprocess it, and use the trained model to predict the genre.


#### HTML Upload Form

Created an index.html file to serve as the main page of the web app. This page includes an upload form where users can select and submit their audio files.

```
<body>
    <div class="container">
        <h1>Music Genre Classifier</h1>
        {% if result %}
            <div class="result">
                Predicted Genre: {{ result }}
            </div>
        {% endif %}
        <form method='POST' enctype='multipart/form-data'>
            {{form.hidden_tag()}}
            <label for="file">{% if result %}Choose another audio file{% else %}Choose an audio file{% endif %}</label>
            {{form.file(id="file", onchange="updateFileName(this)")}}
            <div class="file-name" id="file-name"></div>
            {{form.submit()}}
        </form>
    </div>

    <script>
        function updateFileName(input) {
            const fileName = input.files[0].name;
            document.getElementById('file-name').textContent = fileName;
        }
    </script>
</body>
```

#### Running the application using Flask
In app.py, define the Flask routes and the prediction logic. The /home route handles file upload, processes the audio file, and predicts the genre using the trained model.

```

@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    result = None
    if form.validate_on_submit():
        file = form.file.data
        folder = "test_song"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["UPLOAD_FOLDER"],
                secure_filename(file.filename),
            )
        )
        # run the neural network here
        result = test.run()
    return render_template("index.html", form=form, result=result)

```


Visit http://127.0.0.1:5000/ in your web browser to access the web app.



### Future Enhancements
- Error Handling: Implement more robust error handling for unsupported file types or corrupted audio files.
- Enhanced UI/UX: Improve the user interface with more detailed prediction results and visualizations.
- Deployment: Deploy the Flask app to a cloud platform like Heroku or AWS for broader accessibility.


https://github.com/user-attachments/assets/d30641c3-1820-4249-83ef-197d17e9ea00


## Future

I would like to further refine the model by optimizing hyperparameters, experimenting with different RNN variants, and potentially incorporating data augmentation techniques to enhance the dataset's diversity.

## Acknowledgements

A special thank you to Valerio Velardo and his incredible YouTube course, [Deep Learning (for Audio) with Python](https://www.youtube.com/playlist?list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf). His tutorials and insights have been instrumental in the development of this music genre classifier project.

## License

This project is licensed under the MIT License.
