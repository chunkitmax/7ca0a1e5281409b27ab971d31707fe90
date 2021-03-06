'''
train.py
train model
'''

import gzip
import os
import pickle
import re
import shutil
import sys
from collections import deque
from zipfile import ZipFile

import numpy as np
import torch as T

from dataloader import DataManager
from log import Logger
from RNN import RNN_M2M, RNN_M2O


class Trainer:
  def __init__(self, is_many_to_one=True, max_epoch=5000, batch_size=10,
               learning_rate=.01, hidden_size=128, num_hidden_layer=1,
               drop_rate=0., embedding_len=100, use_tensorboard=False,
               early_stopping_history_len=7, early_stopping_allowance=3,
               verbose=1, save_best_model=False, use_cuda=False,
               data_file_count=-1, identity=None, early_stopping=False,
               pre_train=None):
    self.logger = Logger(verbose_level=verbose)
    self.is_many_to_one = is_many_to_one
    self.max_epoch = max_epoch
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.hidden_size = hidden_size
    self.num_hidden_layer = num_hidden_layer
    self.drop_rate = drop_rate
    self.embedding_len = embedding_len
    self.use_cuda = use_cuda
    self.use_tensorboard = use_tensorboard
    self.early_stopping_history_len = early_stopping_history_len
    self.early_stopping_allowance = early_stopping_allowance
    self.verbose = verbose
    self.save_best_model = save_best_model
    self.data_file_count = data_file_count
    self.identity = identity
    self.early_stopping = early_stopping
    self.pre_train = pre_train
  def train(self):
    data_manager = DataManager(self.batch_size, logger=self.logger,
                               is_many_to_one=self.is_many_to_one,
                               data_file_count=self.data_file_count,
                               pretrained_file=self.pre_train)
    if self.is_many_to_one:
      net = RNN_M2O(len(data_manager.word_list), self.embedding_len,
                    self.hidden_size, self.learning_rate, self.num_hidden_layer,
                    self.drop_rate, use_adam=True, use_cuda=self.use_cuda,
                    pretrained_emb=data_manager.pretrained_embeddings())
    else:
      net = RNN_M2M(len(data_manager.word_list), self.embedding_len,
                    self.hidden_size, self.learning_rate, self.num_hidden_layer,
                    self.drop_rate, use_adam=True, use_cuda=self.use_cuda,
                    pretrained_emb=data_manager.pretrained_embeddings())
    self._train(net, data_manager)
  def _train(self, net, data_manager):
    if self.identity is None:
      identity = 'M2O' if self.is_many_to_one else 'M2M'
      identity += '_'+str(self.learning_rate).replace('.', '')
      identity += '_'+str(self.hidden_size)
      identity += '_'+str(self.num_hidden_layer)
    else:
      identity = self.identity
    if self.use_tensorboard:
      from tensorboardX import SummaryWriter
      if os.path.exists(identity+'_logs'):
        if self.verbose > 0:
          should_rm = input(' - Log dir exists. Remove (Y/n)?')
          if should_rm.lower() == 'y' or should_rm == '':
            shutil.rmtree(identity+'_logs')
      self.writer = SummaryWriter(identity+'_logs')

    train_data_loader = data_manager.train_loader()
    valid_data_loader = data_manager.valid_loader()

    optimizer = net.get_optimizer()
    loss_fn = net.get_loss()
    self.logger.i('Start training %s...'%(identity), True)
    try:
      total_batch_per_epoch = len(train_data_loader)
      perplexity_history = deque(maxlen=self.early_stopping_history_len)
      min_perplexity = 999.
      early_stopping_violate_counter = 0
      status, _epoch_index, _perplexity_history, _min_perplexity = self._load(net, identity)
      if status:
        perplexity_history = _perplexity_history
        min_perplexity = _min_perplexity
      else:
        _epoch_index = 0
      epoch_index = 0
      for epoch_index in range(_epoch_index, self.max_epoch):
        losses = 0.
        acc = 0.
        counter = 0
        self.logger.i('[ %d / %d ] epoch:'%(epoch_index + 1, self.max_epoch), True)
        # Training
        net.train()
        for batch_index, (data, label) in enumerate(train_data_loader):
          data = T.autograd.Variable(data)
          label = T.autograd.Variable(label)
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = net(data)
          acc += (label.squeeze() == predicted).float().mean().data * data.size(0)
          loss = loss_fn(output.view(-1, len(data_manager.word_list)), label.view(-1))
          optimizer.zero_grad()
          loss.backward()
          T.nn.utils.clip_grad_norm(net.parameters(), .25)
          optimizer.step()
          losses += loss.data.cpu()[0] * data.size(0)
          counter += data.size(0)
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, acc: %.4f'%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter, acc / counter))
        mean_loss = losses / counter
        valid_losses = 0.
        valid_counter = 0
        valid_acc = 0.
        # Validtion
        net.eval()
        for data, label in valid_data_loader:
          data = T.autograd.Variable(T.LongTensor(data))
          label = T.autograd.Variable(T.LongTensor(label))
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = net(data)
          valid_losses += loss_fn(output.view(-1, len(data_manager.word_list)), label.view(-1)) \
                                 .data.cpu()[0] * data.size(0)
          valid_acc += (label.squeeze() == predicted).float().mean().data * data.size(0)
          valid_counter += data.size(0)
        mean_val_loss = valid_losses/valid_counter
        mean_val_acc = valid_acc/valid_counter
        perplexity = np.exp(mean_val_loss)
        self.logger.d(' -- val_loss: %.4f, val_acc: %.4f, perplexity: %.4f'%
                      (mean_val_loss, mean_val_acc, perplexity), reset_cursor=False)
        # Log with tensorboard
        if self.use_tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          self.writer.add_scalar('train_acc', acc / counter, epoch_index)
          self.writer.add_scalar('val_loss', mean_val_loss, epoch_index)
          self.writer.add_scalar('val_acc', mean_val_acc, epoch_index)
          self.writer.add_scalar('val_perp', perplexity, epoch_index)
        # Early stopping
        if self.early_stopping and perplexity > np.mean(perplexity_history):
          early_stopping_violate_counter += 1
          if early_stopping_violate_counter >= self.early_stopping_allowance:
            self.logger.i('Early stopping...', True)
            break
        else:
          early_stopping_violate_counter = 0
        # Save best model
        if self.save_best_model and perplexity < min_perplexity:
          self._save(epoch_index, net, perplexity_history, perplexity, identity)
          min_perplexity = perplexity
        perplexity_history.append(perplexity)
        self.logger.d('', True, False)
    except KeyboardInterrupt:
      self.logger.i('\n\nInterrupted', True)
    if self.use_tensorboard:
      self.writer.close()
    self.logger.i('Finish', True)
    return np.mean(perplexity_history)
  def test(self, id):
    _, lr, hs, nh = re.search(r'M2(M|O)_([0-9]+)_([0-9]+)_([0-9]+)_?', id).groups()
    lr, hs, nh = float('0.'+lr[1:]), int(hs), int(nh)

    data_manager = DataManager(self.batch_size, logger=self.logger,
                               is_many_to_one=self.is_many_to_one,
                               data_file_count=self.data_file_count,
                               pretrained_file=self.pre_train, is_test=True)
    if self.is_many_to_one:
      model = RNN_M2O
    else:
      model = RNN_M2M
    net = model(len(data_manager.word_list), self.embedding_len,
                hs, lr, nh, self.drop_rate, use_adam=True, use_cuda=self.use_cuda,
                pretrained_emb=data_manager.pretrained_embeddings())
    status, _epoch_index, _perplexity_history, _min_perplexity = self._load(net, id)
    if status:
      loss_fn = net.get_loss()

      # Testing
      test_losses = 0.
      test_acc = 0.
      test_counter = 0

      net.eval()
      for data, label in data_manager.test_loader():
        data = T.autograd.Variable(T.LongTensor(data))
        label = T.autograd.Variable(T.LongTensor(label))
        if self.use_cuda:
          data = data.cuda()
          label = label.cuda()
        output, predicted = net(data)
        test_losses += loss_fn(output.view(-1, len(data_manager.word_list)), label.view(-1)) \
                                .data.cpu()[0] * data.size(0)
        test_acc += (label.squeeze() == predicted).float().mean().data * data.size(0)
        test_counter += data.size(0)
      mean_test_loss = test_losses/test_counter
      mean_test_acc = test_acc/test_counter
      perplexity = np.exp(mean_test_loss)
      self.logger.i('Loss: %.4f, Acc: %.4f, Perp: %.4f'%(mean_test_loss, mean_test_acc, perplexity))
      return mean_test_loss, mean_test_acc, perplexity
    else:
      raise AssertionError('Model file not found!')
  def text_generate(self, given_words, id, max_len=150):
    if os.path.exists('data/word_list'):
      word_list = pickle.load(open('data/word_list', 'rb'))
    else:
      raise AssertionError('word_list not found')

    _, lr, hs, nh = re.search(r'M2(M|O)_([0-9]+)_([0-9]+)_([0-9]+)_?', id).groups()
    lr, hs, nh = float('0.'+lr[1:]), int(hs), int(nh)

    if self.is_many_to_one:
      net = RNN_M2O(len(word_list), self.embedding_len,
                    hs, lr, nh, self.drop_rate, use_adam=True, use_cuda=self.use_cuda)
    else:
      net = RNN_M2M(len(word_list), self.embedding_len,
                    hs, lr, nh, self.drop_rate, use_adam=True, use_cuda=self.use_cuda)
    status, _, _, _ = self._load(net, id)
    if status:
      word_index_dict = {w: i for i, w in enumerate(word_list)}
      given_words = given_words.lower().strip().split()
      given_words = [1]+[word_index_dict[word] if word in word_index_dict else 2
                         for word in given_words]
      state = None
      for i in range(max_len):
        if i < len(given_words):
          cur_var = T.autograd.Variable(T.LongTensor([[given_words[i]]]))
          if self.use_cuda:
            cur_var = cur_var.cuda()
          _, predicted, state = net(cur_var, state, return_states=True)
          if i >= len(given_words)-1:
            if predicted[0].cpu().data[0] > 0:
              given_words.append(predicted[0].cpu().data[0])
            else:
              break
      print('Text generated: %s'%(' '.join([word_list[word] for word in given_words[1:]])))
      print('Finished')
    else:
      raise AssertionError('Save not found!')
  def _save(self, global_step, net, perplexity_history, min_perplexity, identity):
    T.save({
        'epoch': global_step+1,
        'state_dict': net.state_dict(),
        'perplexity_history': perplexity_history,
        'min_perplexity': min_perplexity,
        'optimizer': net.optimizer.state_dict()
    }, identity+'_best')
  def _load(self, net, identity):
    if os.path.exists(identity+'_best'):
      checkpoint = T.load(identity+'_best')
    elif os.path.exists(identity):
      checkpoint = T.load(identity)
    else:
      return False, None, None, None
    net.load_state_dict(checkpoint['state_dict'])
    net.get_optimizer().load_state_dict(checkpoint['optimizer'])
    return True, checkpoint['epoch'], checkpoint['perplexity_history'], \
           checkpoint['min_perplexity']
