'''
RNN.py
define RNN models
'''

import torch as T

from dataloader import DataManager


class RNN_M2M(T.nn.Module):
  def __init__(self, word_list_len, embedding_len, hidden_size=50, lr=0.01,
               num_layers=3, drop_rate=0., use_cuda=False, use_adam=True,
               use_rmsprop=False):
    super(RNN_M2M, self).__init__()
    self.word_list_len = word_list_len
    self.embedding_len = embedding_len
    self.hidden_size = hidden_size
    self.lr = lr
    self.num_layers = num_layers
    self.drop_rate = drop_rate
    self.use_cuda = use_cuda
    self.use_adam = use_adam
    self.use_rmsprop = use_rmsprop
    self._build_model()
  def _build_model(self):
    self.Embedding = T.nn.Embedding(self.word_list_len, self.embedding_len, padding_idx=0,
                                    scale_grad_by_freq=True)
    self.RNN = T.nn.GRU(input_size=self.embedding_len, hidden_size=self.hidden_size,
                        num_layers=self.num_layers, batch_first=True, dropout=self.drop_rate)
    self.Fc = T.nn.Linear(self.hidden_size, self.word_list_len)
    self.Loss = T.nn.CrossEntropyLoss()
    if self.use_cuda:
      self.cuda()
    if self.use_adam:
      self.optimizer = T.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.lr)
    elif self.use_rmsprop:
      self.optimizer = T.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()),
                                       lr=self.lr)
    else:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=0.9, nesterov=True)
  def forward(self, inputs):
    embeddings = self.Embedding(inputs)
    output, _ = self.RNN(embeddings)
    output = self.Fc(output)
    _, max_indice = T.max(output, dim=2)
    return output, max_indice

  def get_loss(self):
    return self.Loss
  
  def get_optimizer(self):
    return self.optimizer

class RNN_M2O(T.nn.Module):
  def __init__(self, word_list_len, embedding_len, hidden_size=50, lr=0.01,
               num_layers=3, drop_rate=0., use_cuda=False, use_adam=True,
               use_rmsprop=False):
    super(RNN_M2O, self).__init__()
    self.word_list_len = word_list_len
    self.embedding_len = embedding_len
    self.hidden_size = hidden_size
    self.lr = lr
    self.num_layers = num_layers
    self.drop_rate = drop_rate
    self.use_cuda = use_cuda
    self.use_adam = use_adam
    self.use_rmsprop = use_rmsprop
    self._build_model()
  def _build_model(self):
    self.Embedding = T.nn.Embedding(self.word_list_len, self.embedding_len, padding_idx=0,
                                    scale_grad_by_freq=True)
    self.RNN = T.nn.GRU(input_size=self.embedding_len, hidden_size=self.hidden_size,
                        num_layers=self.num_layers, batch_first=True, dropout=self.drop_rate)
    self.Fc = T.nn.Linear(self.hidden_size, self.word_list_len)
    self.Loss = T.nn.CrossEntropyLoss()
    if self.use_cuda:
      self.cuda()
    if self.use_adam:
      self.optimizer = T.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.lr)
    elif self.use_rmsprop:
      self.optimizer = T.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()),
                                       lr=self.lr)
    else:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=0.9, nesterov=True)
  def forward(self, inputs):
    embeddings = self.Embedding(inputs)
    output, _ = self.RNN(embeddings)
    output = output.narrow(1, inputs.size(1)-1, 1).view(-1, self.hidden_size)
    output = self.Fc(output)
    _, max_indice = T.max(output, dim=1)
    return output, max_indice

  def get_loss(self):
    return self.Loss

  def get_optimizer(self):
    return self.optimizer
