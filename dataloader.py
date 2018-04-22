'''
dataloader.py
read data from files &
preprocess data
'''

import math
import os
import pickle
import re
from collections import Counter
from zipfile import ZipFile

import numpy as np
import torch as T
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from log import Logger


class Dataset(Data.Dataset):

  def __init__(self, data, logger, **args):
    self.data = data
    # Logger
    Log = logger

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

class DataManager:
  def __init__(self, batch_size=50, max_seq=30, logger=None, train_valid_ratio=.2, is_test=False):
    self.batch_size = batch_size
    self.logger = Logger(0)
    self.is_test = is_test

    # mkdir data folder
    if not os.path.exists('data'):
      os.mkdir('data')

    # File path
    file_path_prefix = 'Data/Test' if is_test else 'Data/Train'

    self.data = ''
    self.dataset = None
    self.train_dataset, self.valid_dataset = None, None
    self.word_counter = None

    if is_test and os.path.exists('data/test_data'):
      self.dataset = pickle.load(open('data/test_data', 'rb'))
    elif not is_test and os.path.exists('data/train_data') \
                     and os.path.exists('data/valid_data'):
      self.train_dataset = pickle.load(open('data/train_data', 'rb'))
      self.valid_dataset = pickle.load(open('data/valid_data', 'rb'))
    else:
      # Read zip file
      if os.path.exists('hw4_dataset.zip'):
        with ZipFile('hw4_dataset.zip', 'r') as zf:
          file_count = len(zf.filelist)
          self.logger.i('Start loading dataset...')
          for index, f in enumerate(zf.filelist):
            if f.file_size > 0:
              if f.filename.startswith(file_path_prefix):
                text = zf.read(f.filename).decode('utf-8').lower()
                text = text[text.rindex('*end*')+len('*end*'):text.rindex('end')]
                self.data += self._clean_str(text)+'\n'
            self.logger.i('Loading %3d / %3d docs'%(index+1, file_count))
        self.data = [['<sos>']+entry.split(' ') for entry in self.data.split('\n')]
        # Limit sentense len
        def spliter(d):
          for i in range(math.ceil(len(d)/max_seq)):
            yield d[max_seq*i:max_seq*(i+1)]
        for index, entry in enumerate(self.data):
          if len(entry) > max_seq:
            splits = list(spliter(entry))
            self.data[index] = splits[0]
            self.data.extend(splits[1:])
        self.data = list(filter(lambda x: len(x) > 2, self.data))
        if is_test:
          self.dataset = Dataset(self.data, self.logger)
          pickle.dump(self.dataset, open('data/test_data', 'wb+'))
        else:
          train_data, valid_data = train_test_split(self.data,
                                                    test_size=train_valid_ratio,
                                                    random_state=0)
          self.train_dataset = Dataset(train_data, logger)
          self.valid_dataset = Dataset(valid_data, logger)
          pickle.dump(self.train_dataset, open('data/train_data', 'wb+'))
          pickle.dump(self.valid_dataset, open('data/valid_data', 'wb+'))
      else:
        raise AssertionError('hw4_dataset.zip not found')

    if os.path.exists('data/word_list'):
      self.word_list = pickle.load(open('data/word_list', 'rb'))
      return
    elif is_test:
      raise AssertionError('word_list not found')

    self.logger.i('Start counting words..')
    self.word_counter = Counter()
    train_data = [x for sublist in self.train_dataset.data for x in sublist]
    self.word_counter += Counter(train_data)
    # Only keep words in training data
    del self.word_counter['<sos>']
    filtered_word_list = [k for k, v in self.word_counter.items() if v >= 3]
    self.word_list = ['<pad>', '<sos>', '<unk>']+filtered_word_list
    pickle.dump(self.word_list, open('data/word_list', 'wb+'))
    valid_data = [x for sublist in self.valid_dataset.data for x in sublist]
    self.word_counter += Counter(valid_data)
    # Update unknown words for statistics
    self.word_counter += Counter({'<unk>':0})
    self.logger.i('Cleaning words..')
    unk_word_list = list(filter(lambda p: p[0] not in self.word_list, self.word_counter.items()))
    for k, v in unk_word_list:
      del self.word_counter[k]
      self.word_counter['<unk>'] += v
    del self.word_counter['<sos>']
    pickle.dump(self.word_counter, open('data/word_counter', 'wb+'))

  def test_loader(self):
    return Data.DataLoader(self.dataset, self.batch_size, False)

  def train_loader(self):
    return Data.DataLoader(self.train_dataset, self.batch_size, True)

  def valid_loader(self):
    return Data.DataLoader(self.valid_dataset, self.batch_size, False)

  def _clean_str(self, string):
    '''
    Remove noise from input string
    '''
    # Remove newline characters
    string = re.sub(r'\n|\r', ' ', string)
    # Remove footnotes
    string = re.sub(r'\[Footnote [0-9]+\:.+\]', ' ', string)
    string = re.sub(r'\^[0-9]+', ' ', string)
    # Limit character set
    string = re.sub(r'[^A-Za-z0-9,!?\(\)\.\'\`\"]', ' ', string)
    # Let <num> symbolizes numbers
    string = re.sub(r'[0-9]+', ' <num> ', string)
    # Add space around quotes
    string = re.sub(r'( (\'|\") ?)|( ?(\'|\") )', r' \1 ', string)
    # Separate short forms
    string = re.sub(r'(\'s|\'ve|n\'t|\'re|\'d|\'ll|\.|,|!|\?|\(|\))',
                    r' \1 ', string)
    # Remove consecutive space
    string = re.sub(r'\s{2,}', ' ', string)
    # Insert newline characters
    string = re.sub(r'(,|\.|!|\?|;) ([^\'\"])', r'\1\n\2', string)
    # Lower case
    return string.strip()

if __name__ == '__main__':
  dataset = DataManager(50, Logger())
