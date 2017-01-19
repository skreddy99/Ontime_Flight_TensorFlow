""" Objective: Impost MNIST data; and predict if the number is a numeric between 0 (Zero) and 9 (Nine)

Design of the computation modeL:
Input data - Mul layer1_weights - add layer1_Bias - send it through Hidden Layer1 - trigger the Layer_1 activation function
- Mul Layer2_weights - add Layer2_bias - trigger layer_2 activation fucntion - (so on till all the hiden layers) -
compare output to correct value - apply cost function - apply optimizer to minimise cost - repeat the epoch - output layer

users/skreddy/anakonda/envs/tensorflow/lib/python3.5/site-packages/

Types of optimiserss: AdaGrad, Proxinal optimisers, ProxinalAdaGrad optimisers, Stochastic gradient descent optimiser,
ftrl optimizer, adam optimioser, ada delta optimizer, ada grad dual averaging optimizer, momemtum optimizer, RMS Propo optimizer

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

"""
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

"""
# Data sets
FLIGHT_TRAINING = "ontime_flight_arrival_trainingdata.csv"
FLIGHT_TEST = "ontime_flight_arrival_testdata.csv"

print("data sets done")

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FLIGHT_TRAINING,
    target_dtype=np.int32,
    features_dtype=np.float32,
    target_column=0)

print("first Load datasets done")

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FLIGHT_TEST,
    target_dtype=np.int32,
    features_dtype=np.float32,
    target_column=0)

print("Second Load datasets done")

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=18)]


print("specify that all features... done")

"""
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

"""

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/flight_model")

print("Build 3 layer... done")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

print("fit model done")

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

print("accuracy done")

# Classify
new_samples = np.array(
    [[1,1,24,1845,1825,8,69,1954,11697,15304,35,0,197,5,18,2,4,0],
    [1,0,36,1759,1800,17,113,1952,12953,14100,39,0,96,7,0,0,36,0]
    [1,1,66,2105,1940,5,84,2229,11697,10994,68,0,470,2,6,60,0,0]
    [1,0,17,2317,2305,6,65,22,12266,14683,38,0,191,3,0,12,5,0]
    [1,1,18,1215,1200,10,106,1401,11042,11057,66,0,430,7,15,0,3,0]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))

print("Precition done")
