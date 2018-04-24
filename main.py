from train import Trainer
import argparse

parser = argparse.ArgumentParser(description='RNN-text-generation')
parser.add_argument('mode', type=str, help='Many-to-One (M2O) or Many-to-Many (M2M)')
parser.add_argument('-b', '--batch_size', default=50, type=int, help='Batch size')
parser.add_argument('-e', '--epoch', default=5, type=int, help='Number of epoch to train')
parser.add_argument('-emb', '--emb_len', default=100, type=int,
                    help='Embedding length')
parser.add_argument('-g', '--gpu', action='store_true', help='GPU support')
parser.add_argument('-tb', '--tensorboard', action='store_true', help='TensorBoard')
parser.add_argument('-v', '--verbose', default=1, type=int, help='Verbose level')

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
                    batch_size=Args.batch_size, use_tensorboard=Args.tensorboard,
                    use_cuda=Args.gpu, save_best_model=True, verbose=Args.verbose)
  trainer.train()

if __name__ == '__main__':
  main()
