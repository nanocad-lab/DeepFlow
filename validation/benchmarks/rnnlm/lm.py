import tensorflow as tf
import numpy as np
import time
import argparse
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Input, RNN, LSTMCell, LSTM, Bidirectional, Dense, Embedding, Softmax, Activation, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import experimental as mixed_precision


#tf.debugging.set_log_device_placement(True)
def prepare_data(batch_size, seq_len, vocab_size, bpe):
  # Load data
  (x_train, _), (x_test,_) = imdb.load_data(num_words=vocab_size)

  # Pad sequences
  x_train = sequence.pad_sequences(x_train, maxlen=seq_len+1)[:bpe * batch_size]
  x_test = sequence.pad_sequences(x_test, maxlen=seq_len+1)[:bpe * batch_size]
  num_seq = len(x_train)

  print("num_seq: {}".format(num_seq))
  #y_train
  y_train = [x[1:] for x in x_train]
  y_train_encoded = np.zeros((num_seq, seq_len, vocab_size), dtype=int)
  
  for i, y in enumerate(y_train):
    y_train_encoded[i, :, :] = to_categorical(y, num_classes=vocab_size)
  
  #y_test
  y_test = [x[1:] for x in x_test]
  y_test_encoded = np.zeros((num_seq, seq_len, vocab_size), dtype=int)
  for i, y in enumerate(y_test):
    y_test_encoded[i, :, :] = to_categorical(y, num_classes=vocab_size)
  
  #x_train
  x_train = np.array([x[:-1] for x in x_train], dtype=int)
  
  #x_test
  x_test  = np.array([x[:-1] for x in x_test], dtype=int)

  return x_train, y_train_encoded, x_test, y_test_encoded

def build_model(batch_size, hidden_size, seq_len, vocab_size, num_hidden_layers):
  model = Sequential()
  model.add(Embedding(vocab_size, hidden_size, batch_input_shape=[batch_size, seq_len]))
  for _ in range(num_hidden_layers):
    model.add(LSTM(hidden_size, return_sequences=True, unroll=True))
  model.add(TimeDistributed(Dense(vocab_size)))
  model.add(Activation('softmax', dtype='float32'))
  model.compile(
      optimizer="sgd",
      loss='categorical_crossentropy',
      metrics=['categorical_accuracy'])
  return model




def main():
    parser = argparse.ArgumentParser(formatter_class =
                argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default='1024', help="Batch Size")
    parser.add_argument('-d', '--hidden_dim', type=int, required=False, default='512', help="Hidden Dimension")
    parser.add_argument('-s', '--seq_len', type=int, required=False, default='10', help="Seq. Length")
    parser.add_argument('-v', '--vocab_size', type=int, required=False, default='40000', help="Vocab. Size")
    parser.add_argument('-e', '--num_epoch', type=int, required=False, default='1', help="Number of Epochs")
    parser.add_argument('-p', '--batch_per_epoch', type=int, required=False, default='4', help="Number of Batches per Epoch")
    parser.add_argument('-l', '--num_hidden_layers', type=int, required=False, default='1', help="Number of LSTM layers")


    args = parser.parse_args()

    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    seq_len = args.seq_len
    vocab_size = args.vocab_size
    num_epoch = args.num_epoch
    num_hidden_layers = args.num_hidden_layers
    bpe = args.batch_per_epoch


    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    # Profile from batches 10 to 15
    #log_dir = '/mnt/home/newsha/baidu/projects/language_model/lm-sap/log_dir'
    #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
    #                                             profile_batch='10,15')
    
    #with tf.device("GPU:0"):
    x_train, y_train, x_test, y_test = prepare_data(batch_size, seq_len, vocab_size, bpe)
    model = build_model(batch_size, hidden_dim, seq_len, vocab_size, num_hidden_layers)
    model.summary()
    print("Batch: {}, Seq: {}, Hidden: {}, Vocab: {}, Epochs: {}, Batch_per_Epoch: {}, Layers: {}".format(
           batch_size, seq_len, hidden_dim, vocab_size, num_epoch, bpe, num_hidden_layers))
    model.fit(x_train, y_train, epochs=num_epoch, batch_size=batch_size)#, callbacks=[tb_callback]) #, validation_data=(x_test, y_test))
    


if __name__ == "__main__":
    main()
