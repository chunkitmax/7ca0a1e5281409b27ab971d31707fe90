'''
train.py
train model
'''

import gzip
import os
import sys
from collections import deque
from zipfile import ZipFile

import numpy as np
import torch as T

from dataloader import DataManager
from log import Logger
from RNN import RNN_M2M, RNN_M2O


class Trainer:
  def __init__(self, is_manay_to_one=True, max_epoch=5000, batch_size=100,
               embedding_len=100, use_tensorboard=False, early_stopping_history_len=50,
               early_stopping_allowance=5, verbose=1, save_best_model=False,
               use_cuda=False):
    self.logger = Logger(verbose_level=verbose)
    self.is_manay_to_one = is_manay_to_one
    self.max_epoch = max_epoch
    self.batch_size = batch_size
    self.embedding_len = embedding_len
    self.use_cuda = use_cuda
    self.use_tensorboard = use_tensorboard
    self.early_stopping_history_len = early_stopping_history_len
    self.early_stopping_allowance = early_stopping_allowance
    self.save_best_model = save_best_model
    self.counter = 0
  def train(self):
    if self.is_manay_to_one:
      data_manager = DataManager(self.batch_size, logger=self.logger,
                                 is_many_to_one=self.is_manay_to_one)
      net = RNN_M2O(len(data_manager.word_list),
                    self.embedding_len, use_adam=True, use_cuda=self.use_cuda)
      b_perplexity = self._train(net, data_manager, 'M2O')
    else:
      data_manager = DataManager(self.batch_size, logger=self.logger,
                                 is_many_to_one=self.is_manay_to_one)
      net = RNN_M2M(len(data_manager.word_list),
                    self.embedding_len, use_adam=True, use_cuda=self.use_cuda)
      b_perplexity = self._train(net, data_manager, 'M2M')
  def _train(self, net, data_manager, identity=None):
    if identity is None:
      identity = 'Net'+str(self.counter)
      self.counter += 1
    if self.use_tensorboard:
      from tensorboardX import SummaryWriter
      self.writer = SummaryWriter(identity+'_logs')

    train_data_loader = data_manager.train_loader()
    valid_data_loader = data_manager.valid_loader()

    optimizer = net.get_optimizer()
    loss_fn = net.get_loss()
    self.logger.i('Start training %s...'%(identity), True)
    try:
      total_batch_per_epoch = len(train_data_loader)
      perplexity_history = deque(maxlen=self.early_stopping_history_len)
      min_perplexity = 0.
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
          acc += (label.squeeze() == predicted).float().mean().data
          loss = loss_fn(output.view(-1, len(data_manager.word_list)), label.view(-1))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          losses += loss.data.cpu()[0]
          counter += 1
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, acc: %.4f'%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter, acc / counter))
        mean_loss = losses / counter
        valid_losses = 0.
        valid_counter = 0
        # Validtion
        net.eval()
        for data, label in valid_data_loader:
          data = T.autograd.Variable(T.LongTensor(data))
          label = T.autograd.Variable(T.LongTensor(label))
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = net(data)
          valid_losses += loss_fn(output, label.view(-1)).data.cpu()[0] * data.size(0)
          valid_counter += data.size(0)
        mean_val_loss = valid_losses/valid_counter
        perplexity = np.exp(mean_val_loss)
        self.logger.d(' -- val_loss: %.4f, perplexity: %.4f'%
                      (mean_val_loss, perplexity), reset_cursor=False)
        # Log with tensorboard
        if self.use_tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          self.writer.add_scalar('train_acc', acc / counter, epoch_index)
          self.writer.add_scalar('val_loss', mean_val_loss, epoch_index)
          self.writer.add_scalar('val_perp', perplexity, epoch_index)
        # Early stopping
        if perplexity > np.mean(perplexity_history):
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
  # def _test(self, net, data_loader, max_len=None, test_file='test_for_you_guys.csv'):
  #   counter = 1
  #   net.eval()
  #   for data, _ in test_data_loader:
  #     data = T.autograd.Variable(data)
  #     if self.use_cuda:
  #       data = data.cuda()
  #     _, predicted = net(data)
  #     for prediction in predicted.cpu().data.tolist():
  #       sf.write('%s,%s\n'%(raw_data[counter][:-1], [key for key, value in whole_dataset.sentiments.items() if value==prediction][0]))
  #       counter += 1
  #   print('Finished')
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
      net.load_state_dict(checkpoint['state_dict'])
      net.get_optimizer().load_state_dict(checkpoint['optimizer'])
      return True, checkpoint['epoch'], checkpoint['perplexity_history'], \
             checkpoint['min_perplexity']
    else:
      return False, None, None, None

if __name__ == '__main__':
  trainer = Train(use_cuda=True, use_tensorboard=True)
  trainer.train()
