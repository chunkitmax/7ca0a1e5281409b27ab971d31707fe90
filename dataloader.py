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
from preprocess import clean_str


class Dataset(Data.Dataset):
  def __init__(self, data, word_index_dict, is_many_to_one, sentence_len):
    self.data = []
    self.label = []

    data = [[word_index_dict[word]
             if word in word_index_dict
             else 2 for word in entry]
            for entry in data]

    if is_many_to_one:
      for entry in data:
        for i in range(2, len(entry)-1):
          self.data.append(entry[:i]+[0]*(sentence_len-i))
          self.label.append([entry[i]])
    else:
      for entry in data:
        self.data.append(entry[:-1]+[0]*(sentence_len-len(entry)+1))
        self.label.append(entry[1:]+[0]*(sentence_len-len(entry)+1))

  def __getitem__(self, index):
    return T.LongTensor(self.data[index]), T.LongTensor(self.label[index])

  def __len__(self):
    return len(self.data)

class DataManager:
  def __init__(self, batch_size=50, max_seq=30, logger=None,
               is_many_to_one=False, train_valid_ratio=.2, is_test=False):
    self.batch_size = batch_size
    if logger is None:
      self.logger = Logger(0)
    else:
      self.logger = logger
    self.is_test = is_test
    # Reserve for <sos>
    max_seq += 1

    # mkdir data folder
    if not os.path.exists('data'):
      os.mkdir('data')

    # File path
    file_path_prefix = 'Data/Test' if is_test else 'Data/Train'
    file_path_subfix = '_M2O' if is_many_to_one else '_M2M'

    self.data = ''
    self.dataset = None
    self.word_list = None
    self.train_dataset, self.valid_dataset = None, None
    self.word_counter = None

    if is_test and os.path.exists('data/test_data'):
      self.dataset = pickle.load(open('data/test_data', 'rb'))
    elif not is_test and os.path.exists('data/train_data'+file_path_subfix) \
                     and os.path.exists('data/valid_data'+file_path_subfix):
      self.train_dataset = pickle.load(open('data/train_data'+file_path_subfix, 'rb'))
      self.valid_dataset = pickle.load(open('data/valid_data'+file_path_subfix, 'rb'))
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
                self.data += clean_str(text)+'\n'
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
      else:
        raise AssertionError('hw4_dataset.zip not found')

    if not os.path.exists('data/word_list') and is_test:
      raise AssertionError('word_list not found')
    elif os.path.exists('data/word_list'):
      self.word_list = pickle.load(open('data/word_list', 'rb'))
      self.word_index_dict = {w: i for i, w in enumerate(self.word_list)}
      if is_test:
        self.dataset = Dataset(self.data, self.word_index_dict, is_many_to_one, max_seq-1)
        pickle.dump(self.dataset, open('data/test_data', 'wb+'))
        return
    if os.path.exists('data/train_data') and os.path.exists('data/valid_data'):
      train_data = pickle.load(open('data/train_data', 'rb'))
      valid_data = pickle.load(open('data/valid_data', 'rb'))
    else:
      train_data, valid_data = train_test_split(self.data,
                                                test_size=train_valid_ratio,
                                                random_state=0)
      pickle.dump(train_data, open('data/train_data', 'wb+'))
      pickle.dump(valid_data, open('data/valid_data', 'wb+'))
    if self.word_list is None or not os.path.exists('data/word_counter'):
      self.logger.i('Start counting words..')
      self.word_counter = Counter()
      flatten_train_data = [x for sublist in train_data for x in sublist]
      self.word_counter += Counter(flatten_train_data)
      # Only keep words in training data
      del self.word_counter['<sos>']
      filtered_word_list = [k for k, v in self.word_counter.items() if v >= 3]
      self.word_list = ['<pad>', '<sos>', '<unk>']+filtered_word_list
      self.word_index_dict = {w: i for i, w in enumerate(self.word_list)}
      pickle.dump(self.word_list, open('data/word_list', 'wb+'))
      flatten_valid_data = [x for sublist in valid_data for x in sublist]
      self.word_counter += Counter(flatten_valid_data)
      # Update unknown words for statistics
      self.word_counter += Counter({'<unk>':0})
      self.logger.i('Cleaning words..')
      unk_word_list = list(filter(lambda p: p[0] not in self.word_list, self.word_counter.items()))
      for k, v in unk_word_list:
        del self.word_counter[k]
        self.word_counter['<unk>'] += v
      del self.word_counter['<sos>']
      pickle.dump(self.word_counter, open('data/word_counter', 'wb+'))

    if self.train_dataset is None and self.valid_dataset is None:
      self.train_dataset = Dataset(train_data, self.word_index_dict, is_many_to_one, max_seq-1)
      self.valid_dataset = Dataset(valid_data, self.word_index_dict, is_many_to_one, max_seq-1)
      pickle.dump(self.train_dataset, open('data/train_data'+file_path_subfix, 'wb+'))
      pickle.dump(self.valid_dataset, open('data/valid_data'+file_path_subfix, 'wb+'))

  def test_loader(self):
    return Data.DataLoader(self.dataset, self.batch_size, False)

  def train_loader(self):
    return Data.DataLoader(self.train_dataset, self.batch_size, True)

  def valid_loader(self):
    return Data.DataLoader(self.valid_dataset, self.batch_size, False)
