import tensorflow as tf
import numpy as np
import time
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import shutil
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Input, RNN, LSTMCell, LSTM, Dense, Embedding, Softmax, Activation, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime

from nvtx.plugins.tf.keras.layers import NVTXStart, NVTXEnd
from nvtx.plugins.tf.keras.callbacks import NVTXCallback


#tf.debugging.set_log_device_placement(True)
#def load_data(vocab_size):
#  print("================")
#  print("Loading Data")
#  print("================")
#  (x_train, _), (x_test, _) = imdb.load_data(num_words=vocab_size)
#  return x_train, x_test


def read_words(filename):
    if os.path.isfile(filename):
      with open(filename, "r") as f:
        lines = f.readlines()
    else:
      print("ERROR: {} does not exist".format(filename))
      exit(0)

    return [line.rstrip('\n') for line in lines]

def file_to_word_ids(filename, window_len=20, vocab_size=40000, start_char=1, oov_char=2, index_from=3):
    indexed_data = []
    data = read_words(filename)

    for line in data:
      tokens = [int(w) for w in line.strip('\n').split(" ")]
      if len(tokens) >= window_len:
        tokens = tokens[:window_len]
        tokens = [start_char] + [w + index_from for w in tokens]
        tokens = [wid if wid < vocab_size else oov_char for wid in tokens]
        tokens = np.array(tokens)

        indexed_data.append(tokens)

    indexed_data = np.array(indexed_data, dtype=np.int32)
    return indexed_data


def load_data(window_len, vocab_size, file_path):
    print("=================================================================")
    print("Loading Data: {}".format(file_path))
    print("=================================================================")
    
    return file_to_word_ids(file_path, window_len, vocab_size)


def makedir(output_dir):
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(output_dir)
  print("Created {}".format(output_dir))


def convert_to_text(data, lb=0, ub=1, name=None):
    print("-------------------{}-----------------".format(name))
    word_to_index = imdb.get_word_index()
    for (k,v) in word_to_index.items():
      word_to_index[k] = v + 3

    index_to_word = {}
    index_to_word[0] = "PAD"
    index_to_word[1] = "START"
    index_to_word[2] = "OOV"
    index_to_word = {}
    for (k,v) in word_to_index.items():
      index_to_word[v] = k
    for sentence in data[lb:ub]:
      for wid in sentence:
        if wid in index_to_word:
          print("{}: {}".format(wid, index_to_word[wid]))

def split_feat_label(chunk):
    feature_seq = chunk[:-1]
    label_seq   = chunk[1:]
    #for i, s  in enumerate(tmp_y):
    #  y[i,:,:] = to_categorical(s, num_classes=self.vocab_size)
    return feature_seq, label_seq

def prepare_data(data, window_len, batch_size):
    raw_data    = tf.data.Dataset.from_tensor_slices(data)
    #trim_data   = raw_data.batch(window_len + 1, drop_remainder=True)
    dataset     = raw_data.map(split_feat_label)
    dataset     = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    for input_example_batch, target_example_batch in dataset.take(1):
      print(input_example_batch.shape, target_example_batch.shape)
    return dataset

class BatchGenerator(object):
    def __init__(self, data, window_len, batch_size, vocab_size):
        self.data       = data #sequence.pad_sequences(data, maxlen=window_len+1)
        self.num_steps  = window_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.id = 0

    def generate(self):
        x = np.empty((self.batch_size, self.num_steps))
        y = np.empty((self.batch_size, self.num_steps, self.vocab_size))
        #x = tf.Variable(tf.zeros(shape=(self.batch_size, self.num_steps), dtype=tf.dtypes.int32, name='x'))
        #y = tf.Variable(tf.zeros(shape=(self.batch_size, self.num_steps, self.vocab_size), dtype=tf.dtypes.int32, name='y-hot'))
        while True:
          if self.id + self.batch_size >= len(self.data):
            self.id = 0
          tmp_x = self.data[self.id : self.id + self.batch_size]
          tmp_y = [s[1:] for s in tmp_x]  
          x      = np.array([s[:-1] for s in tmp_x], dtype=np.int32)
          for i, s  in enumerate(tmp_y):
            y[i,:,:] = to_categorical(s, num_classes=self.vocab_size)

          self.id += self.batch_size
          yield x, y


def build_model(batch_size, hidden_size, window_len, vocab_size, num_hidden_layers):
    print("=================================================================")
    print("Building Model")
    print("=================================================================")
    #strategy = tf.distribute.MirroredStrategy()
    #num_gpus = strategy.num_replicas_in_sync
    #print('Number of devices: {}'.format(num_gpus))
    #with strategy.scope():
    inputs = Input(shape=(window_len,))
    x = inputs
    x, marker_id, domain_id = NVTXStart(message='embedding', domain_name='forward', trainable=True)(x)
    x = Embedding(vocab_size, hidden_size)(x) #, batch_input_shape=[batch_size, window_len])(x)
    x = NVTXEnd(grad_message='embedding grad', grad_domain_name='backward')([x, marker_id, domain_id])
    
    for i in range(num_hidden_layers):
      x, marker_id, domain_id = NVTXStart(message='lstm{}'.format(i), domain_name='forward')(x)
      x = LSTM(hidden_size, return_sequences=True, unroll=True)(x)
      x = NVTXEnd(grad_message='lstm{} grad'.format(i), grad_domain_name='backward')([x, marker_id, domain_id])
    #model.add(Dropout(0.5))
    
    x, marker_id, domain_id = NVTXStart(message='projection', domain_name='forward')(x)
    x = TimeDistributed(Dense(vocab_size))(x)
    x = NVTXEnd(grad_message='projection grad', grad_domain_name='backward')([x, marker_id, domain_id])
    
    x, marker_id, domain_id = NVTXStart(message='softmax', domain_name='forward')(x)
    x = Activation('softmax', dtype='float32')(x)
    x = NVTXEnd(grad_message='softmax grad', grad_domain_name='backward')([x, marker_id, domain_id])
    
    predictions = x
    model = Model(inputs = inputs, outputs = predictions)
    model.compile(
        optimizer="sgd",
        #optimizer="adam", #tf.keras.optimizers.Adam(learning_rate=0.001 * np.sqrt(num_gpus)),
        loss= 'sparse_categorical_crossentropy')#tf.keras.losses.CategoricalCrossentropy(from_logits=True)) #'categorical_crossentropy')
    return model



def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64, help="Batch Size")
    parser.add_argument('-d', '--hidden_dim', type=int, required=False, default=4096, help="Hidden Dimension")
    parser.add_argument('-s', '--window_len', type=int, required=False, default=10, help="Seq. Length")
    parser.add_argument('-v', '--vocab_size', type=int, required=False, default=40000, help="Vocab. Size")
    parser.add_argument('-e', '--num_epoch', type=int, required=False, default=1, help="Number of Epochs")
    parser.add_argument('-p', '--batch_per_epoch', type=int, required=False, default=None, help="Number of Batches per Epoch")
    parser.add_argument('-l', '--num_hidden_layers', type=int, required=False, default=1, help="Number of LSTM layers")
    parser.add_argument('-m', '--mode', required=True, help="Train or test")
    parser.add_argument('-c', '--checkpoint_dir', required=False, default='checkpoints' , help="path to your checkpoint directory")
    parser.add_argument('-train', '--indexed_train', required=True , help="path to your indexed train file")
    parser.add_argument('-test', '--indexed_test', required=True , help="path to your indexed test file")
    parser.add_argument('-valid', '--indexed_valid', required=True , help="path to your indexed validation file")


    args = parser.parse_args()

    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    window_len = args.window_len
    vocab_size = args.vocab_size
    num_epoch = args.num_epoch
    num_hidden_layers = args.num_hidden_layers
    bpe = args.batch_per_epoch
    mode = args.mode
    checkpoint_dir = args.checkpoint_dir
    train_file = args.indexed_train
    test_file = args.indexed_test
    valid_file = args.indexed_valid


    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)
    #print('Compute dtype: %s' % policy.compute_dtype)
    #print('Variable dtype: %s' % policy.variable_dtype)


    t0 = time.time() 
    model = build_model(batch_size, hidden_dim, window_len, vocab_size, num_hidden_layers)
    model.summary()
    t1 = time.time() 
    print("Time: {} sec.".format(t1-t0))
 
    train_data = load_data(window_len, vocab_size, train_file)
    t2 = time.time() 
    print("Time: {} sec.".format(t2-t1))
    
    #test_data  = load_data(window_len, vocab_size, test_file)
    #t3 = time.time() 
    #print("Time: {} sec.".format(t3-t2))
    #
    #valid_data = load_data(window_len, vocab_size, valid_file)
    #t4 = time.time() 
    #print("Time: {} sec.".format(t4-t3))
    
    best_valid_file = '{}/best.txt'.format(checkpoint_dir)

    if mode == 'train':
      train_dataset           = prepare_data(train_data, window_len,  batch_size)
      
      #train_data_generator   = BatchGenerator(train_data, window_len,  batch_size, vocab_size)
      if bpe == None:
        bpe = len(train_data)//(batch_size)
      #valid_data_generator   = BatchGenerator(valid_data, window_len, batch_size, vocab_size)
      #checkpointer = ModelCheckpoint(filepath=checkpoint_dir + '/model-{epoch:02d}.hdf5', verbose=1, save_weights_only=True) #, save_freq=5)
      log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

      tb_callback = NVTXCallback()

      makedir(checkpoint_dir)
      print("================")
      print("Training...")
      print("================")
      train_history  = model.fit(x=train_dataset, steps_per_epoch=bpe, epochs=num_epoch, shuffle=False, callbacks=[tb_callback])

      #train_history = model.fit(x = train_data_generator.generate(), steps_per_epoch=bpe, 
      #                          epochs=num_epoch, shuffle=False)
                                #validation_data=valid_data_generator.generate(),
                                #validation_steps=len(valid_data)//(batch_size), callbacks=[checkpointer])

      #val_loss = train_history.history['val_loss']
      #best_epoch, best_val = val_loss.index(min(val_loss)), min(val_loss)
      #with open(best_valid_file, 'w') as f:
      #  f.write('{}: {}'.format(best_epoch + 1, best_val))
    
    elif mode == 'test':
      test_data_generator   = BatchGenerator(test_data, window_len, batch_size, vocab_size)
      with open(best_valid_file, 'r') as f:
        best_epoch_id = int(f.readline().strip().split(':')[0])
        best_model = '{}/model-{:02}.hdf5'.format(checkpoint_dir, best_epoch_id)
        model.load_weights(best_model)
        print("================")
        print("Testing...")
        print("================")
        loss, acc = model.evaluate(x=test_data_generator.generate(), steps=len(test_data)//batch_size, verbose=1)
        print("loss: {}, acc: {}".format(loss, acc))
    
    elif mode == 'predict':
        example_test_generator   = BatchGenerator(test_data, window_len, batch_size, vocab_size)
        with open(best_valid_file, 'r') as f:
          best_epoch_id = int(f.readline().strip().split(':')[0])
          best_model = '{}/model-{:02}.hdf5'.format(checkpoint_dir, best_epoch_id)
          model.load_weights(best_model)
        print("================")
        print("Genrating...")
        print("================")
        gen_data = []
        for i in range(1): #len(test_data)//batch_size):
          data = next(example_test_generator.generate())
          prediction = model.predict(data[0])
          batch=[]
          for j in range(batch_size):
            sentence = []
            for k in range(window_len):
              sentence.append(np.argmax(prediction[j, k, :]))
            batch.append(sentence)
          gen_data.append(batch)

          convert_to_text(gen_data[0], 2,4, "gen_data:")
          convert_to_text(data[0][1:], 2,4, "baseline")
        

if __name__ == "__main__":
    main()
