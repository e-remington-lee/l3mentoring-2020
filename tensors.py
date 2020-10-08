import numpy as np


x = np.array(1)
y = np.array((1,2))

print(x.ndim)
print(y.ndim)

x = [{"a" : 1}, {"c" : 1}, {"b":2}]
print(x)
x = [1,2,3,4,5]
print(x[0:2])
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import json
import bert
import tqdm
from bert.tokenization.bert_tokenization import FullTokenizer
max_sequence_length=192
bert_model = tf.saved_model.load("./bert_en_uncased_L-12_H-768_A-12_2")
input_layer = keras.layers.Input(shape=(max_sequence_length, ), dtype=tf.int32, name="input_layer")
input_mask = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name="segment_ids")
bert_layer = hub.KerasLayer(bert_model, trainable=True)

bert_output = bert_layer([input_layer])