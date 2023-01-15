from tensorflow import keras
import tensorflow as tf
import csv
import pathlib

def saveModel(model):
    
    model.save(input("Save model as: "))

# Code for directory input and model influenced from tensorflow tutorials (https://www.tensorflow.org/tutorials/load_data/images)
def createModel():

    # importing directory
    
    data_dir = "train/"
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 256
    img_width = 256

    x_train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
        )

    x_val = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
        )
    
    num_classes = 6

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    model.fit(x_train, epochs=10, batch_size=8)

    test_loss, test_acc = model.evaluate(x_train, verbose=2)
    print('\n\nTest accuracy: ', test_acc)
    print('\nTest Loss: ', test_loss)

    saveModel(model)


def runModel():
    createModel()

runModel()


