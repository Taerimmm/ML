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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same', input_shape=(28,28)))
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=8, factor=0.8, mode='auto')
    model.fit(x_train, y_train, epochs=3000, batch_size=32, validation_data=(x_test, y_test), callbacks=[es, lr])

    print('Acc :', model.evaluate(x_test, y_test)[1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_certificate/Category2/mymodel.h5")
