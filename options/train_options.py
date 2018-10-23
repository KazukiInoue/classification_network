from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--epochs', type=int, default=100,
                            help='number of epochs to train (default: 50)')
        self.parser.add_argument('--save_latest_freq', type=int, default=200,
                                 help='frequency of saving the latest results (default: 20)')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--runs_dir', type=str, default='./runs',
                                 help='where to save tensorboard setting')
        self.is_train = True