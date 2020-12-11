import tensorflow as tf
import numpy as np
from hparams import hparams
import io

class Feeder():

  def create_dataset(self, paths = "example", num_examples = -1):
    """Create a dataset with all the paths passed"""
    paths = ["D:\\Nacho\\Facultad\\Proyecto Final\\Nachotron\\datasets\\es_ES\\by_book\\female\\karen_savage\\angelina\\metadata.csv"] if paths == "example"  else paths
    lines = []
    for path in paths:
      for line in io.open(path, encoding='UTF-8').read().strip().split("\n"):
        lines.append(line)
    sentences = np.array([line.split('|')[1] for line in lines[:num_examples]])
    audio_identifier = np.array([line.split('|')[0] for line in lines[:num_examples]])
    letters = sorted(set( ''.join(sentences).lower() ))
    self.char2idx = {u:i for i, u in enumerate(letters)}
    self.idx2char = np.array(letters)
    sentences = self.pad_sequences(sentences)
    return sentences, audio_identifier

  def pad_sequences(self, sequences, pad=' '):
    max_len_string = len(max(sequences, key=len))
    self.max_len_string = max_len_string
    return np.array([string.ljust(max_len_string, pad) for string in sequences])

  def vectorize_string(self, string):
    """Returns an array with the idx of each char"""
    return np.array([self.char2idx[char] for char in string.lower()])
  
  def get_batch(self, dataset, batch_size):
    """ Do this """
    sentences, audio_identifier = dataset
    number_of_examples = sentences.shape[0]
    indexes = np.random.choice(number_of_examples, batch_size)
    input_batch = np.array([self.vectorize_string(sentences[idx]) for idx in indexes])
    print(input_batch[0].shape)
    input_batch = np.reshape(input_batch, (batch_size, self.max_len_string))
    output_batch = np.array([audio_identifier[idx] for idx in indexes])
    return input_batch, output_batch

if __name__ == "__main__":
  print("Nachotron feeder test:")
  feeder = Feeder()
  sentences, audio_identifier = feeder.create_dataset()
  print(f'\nSentence 1 "{sentences[0]}" tittle 1 "{audio_identifier[0]}"') 
  vectorized_string = feeder.vectorize_string("My name is Nacho")
  print(f'\nVectorized string: {vectorized_string}')
  print(f'\nchar2idx: {feeder.char2idx}')
  input_batch, output_batch = feeder.get_batch((sentences, audio_identifier), hparams['batch_size'])
  print('\nBatches:\n')
  print(input_batch.shape)
  print(input_batch)
  print(output_batch)