from collections import Counter
import pickle

from dataloader import DataManager
import numpy as np

if __name__ == '__main__':
  data_manager = DataManager()
  data = data_manager.train_dataset.data
  if data_manager.word_counter is None:
    word_counter = pickle.load(open('data/word_counter', 'rb'))
  else:
    word_counter = data_manager.word_counter

  print('Number of sentences: %d'%(len(data)))
  print('Number of words: %d'%(sum(word_counter.values())))
  print('Number of unique words: %d(w/ min freq 3) %d(w/o min freq)'
        %(len([k for k, v in word_counter.items() if v >= 3]),
          len(word_counter)))
  unk_rate = word_counter['<unk>']/sum(word_counter.values())*100.
  print('Coverage of your limited vocabulary: %.2f%%, UNK token rate: %.2f%%'
        %(100.-unk_rate, unk_rate))
  print('Top 10 most frequent words: ', word_counter.most_common(10))
  len_list = [len(line) for line in data]
  print('Maximum sentence length: %d'%(np.max(len_list)))
  print('Minimum sentence length: %d'%(np.min(len_list)))
  print('Average sentence length: %.2f'%(np.mean(len_list)))
  print('Sentence length variation: %.2f'%(np.std(len_list)))
  # print('Distribution of classes: ', class_counter)
