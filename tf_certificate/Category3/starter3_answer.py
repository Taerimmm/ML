# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, './tf_certificate/Category3/rps.zip')
    local_zip = './tf_certificate/Category3/rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('./tf_certificate/Category3/tmp/')
    zip_ref.close()


    TRAINING_DIR = "./tf_certificate/Category3/tmp/rps/"
    training_datagen = ImageDataGenerator( # YOUR CODE HERE)
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = training_datagen.flow_from_directory(
            './tf_certificate/Category3/tmp/rps',
            target_size=(150,150),
            subset='training'
        )
    valid_generator = training_datagen.flow_from_directory(
            './tf_certificate/Category3/tmp/rps',
            target_size=(150,150),
            subset='validation'
        )


    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=8, factor=0.8, mode='auto')
    model.fit(train_generator, epochs=3000, validation_data=valid_generator, batch_size=32, callbacks=[es,lr])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_certificate/Category3/mymodel.h5")
