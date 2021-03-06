import argparse
import os

from train import Trainer

parser = argparse.ArgumentParser(description='RNN-text-generation')
parser.add_argument('mode', type=str, help='Many-to-One (M2O) or Many-to-Many (M2M)')

parser.add_argument('-t', '--test', action='store_true', help='Test model')
parser.add_argument('-tg', '--text_generate', action='store_true', help='Generate text')

parser.add_argument('-es', '--early_stopping', action='store_true', help='Enable early stopping')
parser.add_argument('-g', '--gpu', action='store_true', help='GPU support')
parser.add_argument('-tb', '--tensorboard', action='store_true', help='TensorBoard')

parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size')
parser.add_argument('-e', '--epoch', default=5, type=int, help='Number of epoch to train')
parser.add_argument('-emb', '--emb_len', default=100, type=int, help='Embedding length')
parser.add_argument('-lr', '--learning_rate', default=.01, type=float, help='Learning rate')
parser.add_argument('-hs', '--hidden_size', default=128, type=int, help='Hidden layer size')
parser.add_argument('-nh', '--num_hidden', default=1, type=int, help='Number of hidden layers')
parser.add_argument('-dr', '--drop_rate', default=0., type=float, help='Dropout drop rate')
parser.add_argument('-v', '--verbose', default=1, type=int, help='Verbose level')
parser.add_argument('-dfc', '--data_file_count', default=-1, type=int,
                    help='Max number of data file used for generating training set')

parser.add_argument('-pt', '--pre_train', default=None, type=str, help='Pre-trained embedding file')
parser.add_argument('-id', '--identity', default=None, type=str, help='Model identity')

Args = parser.parse_args()

def main():
  if Args.mode.upper() == 'M2O':
    is_many_to_one = True
  elif Args.mode.upper() == 'M2M':
    is_many_to_one = False
  else:
    parser.print_help()
    return
  trainer = Trainer(is_many_to_one, embedding_len=Args.emb_len, max_epoch=Args.epoch,
                    learning_rate=Args.learning_rate, hidden_size=Args.hidden_size,
                    num_hidden_layer=Args.num_hidden, batch_size=Args.batch_size,
                    drop_rate=Args.drop_rate, use_tensorboard=Args.tensorboard,
                    use_cuda=Args.gpu, save_best_model=True,
                    verbose=Args.verbose, data_file_count=Args.data_file_count,
                    identity=Args.identity, early_stopping=Args.early_stopping,
                    pre_train=Args.pre_train)
  if Args.text_generate:
    start_text = input('Text starts with: ')
    file_list = os.listdir()
    file_list = list(filter(lambda x: x.endswith('_best'), file_list))
    selected_index = input('Please select saved model:\n*'+ \
                          '\n'.join([': '.join(line)
                                     for line in zip([str(x) for x in list(range(len(file_list)))],
                                                     file_list)])+'\nIndex: ')
    try:
      selected_index = int(selected_index)
    except ValueError:
      selected_index = 0
    trainer.text_generate(start_text, file_list[selected_index])
  elif Args.test:
    file_list = os.listdir()
    file_list = list(filter(lambda x: x.endswith('_best'), file_list))
    selected_index = input('Please select saved model:\n*'+ \
                          '\n'.join([': '.join(line)
                                     for line in zip([str(x) for x in list(range(len(file_list)))],
                                                     file_list)])+'\nIndex: ')
    try:
      selected_index = int(selected_index)
    except ValueError:
      selected_index = 0
    trainer.test(file_list[selected_index])
  else:
    trainer.train()

if __name__ == '__main__':
  main()
