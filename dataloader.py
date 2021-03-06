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

from collections import deque
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
      print(self.data[0])

  def __getitem__(self, index):
    return T.LongTensor(self.data[index]), T.LongTensor(self.label[index])

  def __len__(self):
    return len(self.data)

class DataManager:
  def __init__(self, batch_size=50, max_seq=7, logger=None,
               is_many_to_one=False, train_valid_ratio=.2, is_test=False,
               data_split_mode='window', data_file_count=-1, pretrained_file=None):
    self.batch_size = batch_size
    self.max_seq = max_seq
    if logger is None:
      self.logger = Logger(0)
    else:
      self.logger = logger
    self.is_test = is_test
    self.data_split_mode = data_split_mode
    if self.data_split_mode not in ['window', 'sentence']:
      raise AssertionError('unknown split mode')
    self.data_file_count = data_file_count
    self.pretrained_file = pretrained_file

    # Reserve for <sos>
    self.max_seq += 1

    # mkdir data folder
    if not os.path.exists('data'):
      os.mkdir('data')

    # File path
    self.file_path_subfix = '_M2O' if is_many_to_one else '_M2M'
    self.file_path_prefix = 'Data/Test' if self.is_test else 'Data/Train'

    self.data = ''
    self.dataset = None
    self.word_list = None
    self.train_dataset, self.valid_dataset = None, None
    self.word_counter = None

    self.tensors = None
    if self.pretrained_file is not None:
      self.tensors = T.Tensor(self._load_from_pretrain())
      self.word_index_dict = {w: i for i, w in enumerate(self.word_list)}
      is_wordlist_loaded = True
    else:
      is_wordlist_loaded = self._load_wordlist()

    if self.is_test:
      if not self._load_dataset():
        self._read_files()
        self.dataset = Dataset(self.data, self.word_index_dict, is_many_to_one, self.max_seq-1)
        pickle.dump(self.dataset, open('data/test_data'+self.file_path_subfix, 'wb+'))
    else:
      if not self._load_dataset():
        # Load previous split data
        status, train_data, valid_data = self._load_data()
        if not status: # No split data found
          self._read_files()
          train_data, valid_data = train_test_split(self.data,
                                                    test_size=train_valid_ratio,
                                                    random_state=0)
          pickle.dump(train_data, open('data/train_data', 'wb+'))
          pickle.dump(valid_data, open('data/valid_data', 'wb+'))
      if (not is_wordlist_loaded or not os.path.exists('data/word_counter')) \
            and self.pretrained_file is None:
        # Generate word list
        self.logger.i('Start counting words because word list or word counter not found...')
        self.word_counter = Counter()
        flatten_train_data = [x for sublist in train_data for x in sublist]
        self.word_counter += Counter(flatten_train_data)
        # Only keep words in training data
        del self.word_counter['<sos>']
        # Set min freq
        # filtered_word_list = [k for k, v in self.word_counter.items() if v >= 3]
        # self.word_list = ['<pad>', '<sos>', '<unk>']+filtered_word_list
        self.word_list = ['<pad>', '<sos>', '<unk>']+list(self.word_counter.keys())
        self.word_index_dict = {w: i for i, w in enumerate(self.word_list)}
        # Save word list
        pickle.dump(self.word_list, open('data/word_list', 'wb+'))

        # Count words for statistics
        flatten_valid_data = [x for sublist in valid_data for x in sublist]
        self.word_counter += Counter(flatten_valid_data)
        # Update unknown words for statistics
        self.word_counter += Counter({'<unk>':0})
        self.logger.i('Getting unknown word list...')
        unk_word_list = list(filter(lambda p: p[0] not in self.word_list,
                                    self.word_counter.items()))
        self.logger.i('Start deleting words in validation set but not in training set...')
        unk_word_list_len = len(unk_word_list)
        for index, [k, v] in enumerate(unk_word_list):
          del self.word_counter[k]
          self.word_counter['<unk>'] += v
          self.logger.i('Deleting... %5d / %5d'%(index+1, unk_word_list_len))
        del self.word_counter['<sos>']
        # Save word counter
        pickle.dump(self.word_counter, open('data/word_counter', 'wb+'))

        self.logger.i('Finish building word list and word counter')

      if self.train_dataset is None and self.valid_dataset is None:
        # Save training and validation dataset
        self.train_dataset = Dataset(train_data, self.word_index_dict,
                                     is_many_to_one, self.max_seq-1)
        self.valid_dataset = Dataset(valid_data, self.word_index_dict,
                                     is_many_to_one, self.max_seq-1)
        pickle.dump(self.train_dataset, open('data/train_data'+self.file_path_subfix, 'wb+'))
        pickle.dump(self.valid_dataset, open('data/valid_data'+self.file_path_subfix, 'wb+'))

      self.logger.i('Finish Generating training set and validation set')
  def _load_dataset(self):
    if self.is_test and os.path.exists('data/test_data'+self.file_path_subfix):
      self.dataset = pickle.load(open('data/test_data'+self.file_path_subfix, 'rb'))
    elif not self.is_test and os.path.exists('data/train_data'+self.file_path_subfix) \
         and os.path.exists('data/valid_data'+self.file_path_subfix):
      self.train_dataset = pickle.load(open('data/train_data'+self.file_path_subfix, 'rb'))
      self.valid_dataset = pickle.load(open('data/valid_data'+self.file_path_subfix, 'rb'))
    else:
      return False
    self.logger.i('Dataset found!')
    return True
  def _load_wordlist(self):
    if os.path.exists('data/word_list'):
      self.logger.i('Word list found!')
      self.word_list = pickle.load(open('data/word_list', 'rb'))
      self.word_index_dict = {w: i for i, w in enumerate(self.word_list)}
    elif self.is_test:
      raise AssertionError('word_list not found')
    else:
      return False
    return True
  def _load_data(self):
    if os.path.exists('data/train_data') and os.path.exists('data/valid_data'):
      train_data = pickle.load(open('data/train_data', 'rb'))
      valid_data = pickle.load(open('data/valid_data', 'rb'))
      self.logger.i('Training dataset and validation dataset found!')
      return True, train_data, valid_data
    return False, None, None
  def _read_files(self):
    if os.path.exists('hw4_dataset.zip'):
      with ZipFile('hw4_dataset.zip', 'r') as zf:
        if self.data_file_count < 0:
          file_count = len(zf.filelist)
        else:
          file_count = self.data_file_count
        self.logger.i('Start loading dataset...')
        valid_file_counter = 0
        file_list = []
        for f in zf.filelist:
          if f.file_size > 0:
            if f.filename.startswith(self.file_path_prefix):
              text = zf.read(f.filename).decode('utf-8').lower()
              text = text[text.rindex('*end*')+len('*end*'):text.rindex('end')]
              self.data += clean_str(text)+' \n '
              valid_file_counter += 1
              file_list.append(f.filename)
              self.logger.i('Loading %3d docs'%(valid_file_counter))
              if valid_file_counter >= file_count:
                break
        with open('files_used', 'w+') as fu:
          for file_name in file_list:
            fu.write(file_name+'\n')
      if self.data_split_mode == 'window':
        tmp_data = self.data
        self.data = []
        window = deque(maxlen=self.max_seq)
        window.append('<sos>')
        for word in tmp_data.strip().split(' '):
          if word == '\n':
            word = '<sos>'
          window.append(word)
          if len(window) == self.max_seq:
            self.data.append(window.copy())
      else:
        self.data = [['<sos>']+entry.split(' ') for entry in self.data.split('\n')]
        # Limit sentense len
        def spliter(d):
          for i in range(math.ceil(len(d)/self.max_seq)):
            yield d[self.max_seq*i:self.max_seq*(i+1)]
        for index, entry in enumerate(self.data):
          if len(entry) > self.max_seq:
            splits = list(spliter(entry))
            self.data[index] = splits[0]
            self.data.extend(splits[1:])
      self.data = list(filter(lambda x: len(x) > 2, self.data))
    else:
      raise AssertionError('hw4_dataset.zip not found')
  def _load_from_pretrain(self):
    self.logger.i('Loading pre-trained embeddings...')
    if not (os.path.exists('data/pre_trained_word_list') \
            and os.path.exists('data/pre_trained_embeddings')):
      self.word_list = ['<pad>', '<sos>', '<unk>', '<num>']
      tensors = [[], [], [], []]
      special_word_dict = {'<pad>': 0, '<sos>': 1, '<unk>': 2,
                           '<unknown>': 2, '<num>': 3, '<number>': 3}
      is_digit = re.compile(r'^[0-9e\.\-\+]+$')
      is_in_limited_char_set = re.compile(r'^[A-Za-z0-9,!?\(\)\.\'\`\"\-]+$')
      with open(self.pretrained_file, 'r') as pt:
        lines = pt.readlines()
        num_line = len(lines)
        for index, line in enumerate(lines):
          word, *embedding = line.strip().split()
          embedding = [float(value) for value in embedding]
          if len(embedding) < 100: # May be caused by emojis / rear words
            continue
          if word in special_word_dict.keys():
            tensors[special_word_dict[word]] = embedding
          elif (is_digit.search(word) is not None or \
                is_in_limited_char_set.search(word) is not None) \
                  and not word.startswith('<'):
            self.word_list.append(word)
            tensors.append(embedding)
          self.logger.d('Loading pre-trained embeddings %6d / %6d...'%(index, num_line))
      # Check if any special symbol has empty embedding
      for i in range(4):
        if len(tensors[i]) == 0:
          tensors[i] = [0.]*len(tensors[4])
      pickle.dump(self.word_list, open('data/pre_trained_word_list', 'wb+'))
      pickle.dump(tensors, open('data/pre_trained_embeddings', 'wb+'))
    else:
      self.logger.i('Pre trained wordlist and embeddings data found!')
      self.word_list = pickle.load(open('data/pre_trained_word_list', 'rb'))
      tensors = pickle.load(open('data/pre_trained_embeddings', 'rb'))
    return tensors
  def pretrained_embeddings(self):
    return self.tensors
  def test_loader(self):
    return Data.DataLoader(self.dataset, self.batch_size, False)
  def train_loader(self):
    return Data.DataLoader(self.train_dataset, self.batch_size, True)
  def valid_loader(self):
    return Data.DataLoader(self.valid_dataset, self.batch_size, False)
