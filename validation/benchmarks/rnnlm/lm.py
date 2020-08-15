import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Input, LSTM, Bidirectional, Dense, Embedding, Softmax, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import experimental as mixed_precision


#tf.debugging.set_log_device_placement(True)

vocab_size = 1000
seq_len = 20
batch_size = 1024
hidden_size = 1024
epochs = 5

def prepare_data():
  # Load data
  (x_train, _), (x_test,_) = imdb.load_data(num_words=vocab_size)

  # Pad sequences
  x_train = sequence.pad_sequences(x_train, maxlen=seq_len+1)
  x_test = sequence.pad_sequences(x_test, maxlen=seq_len+1)
  num_seq = len(x_train)

  print("num_seq: {}".format(num_seq))
  #y_train
  y_train = [x[1:] for x in x_train]
  y_train_encoded = np.zeros((num_seq, seq_len, vocab_size))
  
  for i, y in enumerate(y_train):
    y_train_encoded[i, :, :] = to_categorical(y, num_classes=vocab_size)
  
  #y_test
  y_test = [x[1:] for x in x_test]
  y_test_encoded = np.zeros((num_seq, seq_len, vocab_size))
  for i, y in enumerate(y_test):
    y_test_encoded[i, :, :] = to_categorical(y, num_classes=vocab_size)
  
  #x_train
  x_train = np.array([x[:-1] for x in x_train])
  #x_test
  x_test  = np.array([x[:-1] for x in x_test])

  return x_train, y_train_encoded, x_test, y_test_encoded

def build_model():
  model = Sequential()
  model.add(Embedding(vocab_size, hidden_size))
  model.add(LSTM(hidden_size, return_sequences=True))
  model.add(LSTM(hidden_size, return_sequences=True))
  model.add(Dense(vocab_size))
  model.add(Activation('softmax', dtype='float32'))
  model.compile(
      optimizer="sgd",
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  return model

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# Profile from batches 10 to 15
#log_dir = '/mnt/home/newsha/baidu/projects/language_model/lm-sap/log_dir'
#tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#                                             profile_batch='10,15')

x_train, y_train, x_test, y_test = prepare_data()
model = build_model()
model.summary()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)#, callbacks=[tb_callback]) #, validation_data=(x_test, y_test))
