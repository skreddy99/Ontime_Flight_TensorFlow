from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
'''
This experiment captured the data about domestic flights that arruve and depart in US. 
The huge data is used to make a prediction on the flight ontime arrival. 
'''
# Data sets
FLIGHT_TRAINING = "ontime_flight_arrival_trainingdata.csv"
FLIGHT_TEST = "ontime_flight_arrival_testdata.csv"


# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FLIGHT_TRAINING,
    target_dtype=np.int32,
    features_dtype=np.float32,
    target_column=0)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FLIGHT_TEST,
    target_dtype=np.int32,
    features_dtype=np.float32,
    target_column=0)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=18)]


# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/flight_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


