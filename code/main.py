from utils.tensor_board import Tensorboard
import os
import torch
import numpy
import random
import argparse
import datetime
from Tarcoco import Trainer
from utils.logger import *
from torch.utils.data import DataLoader
from configs.config import config
from dataloaders.dataset import ACDCDataset
from dataloaders.dataset import TwoStreamBatchSampler


def main(args):
    # update the default config with the args
    config.update(vars(args))

    def worker_init_fn(worker_id):
        random.seed(config.seed+worker_id)

    train_set = ACDCDataset(os.path.join(config.code_path, "Dataloader"),
                            config.data_path,
                            split="train", config=config)

    # merge both labelled & unlabelled sampler to same batch
    """
    transfer patience to slice number
    https://github.com/DeepMed-Lab-ECNU/BCP/blob/a925e3018b23255e65a62dd34ae9ac9fc18c0bc9/code/ACDC_BCP_train.py#L213
    """
    patience_to_slice = {1: 32, 3: 68, 7: 136, 14: 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    batch_sampler = TwoStreamBatchSampler(list(range(patience_to_slice[config.labeled_num])),
                                          list(range(patience_to_slice[config.labeled_num], len(train_set))),
                                          config.batch_size, int(config.batch_size/2))

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                              # (pin_memory=False) to avoid the warning:
                              # "Leaking Caffe2 thread-pool after fork. (function pthreadpool)"
                              num_workers=config.num_workers, pin_memory=False,
                              worker_init_fn=worker_init_fn)

    val_dataset = ACDCDataset(os.path.join(config.code_path, "Dataloader"),
                              config.data_path,
                              split="eval", num=None, config=config)

    config.iter_per_epoch = len(train_loader)
    config.n_epochs = config.max_iterations//len(train_loader)+1
    config.unlabeled_num = len(train_set) - config.labeled_num

    logger = logging.getLogger("TraCoCo")
    logger.propagate = False
    logger.info("training with {} epochs [{} iters]".format(config.n_epochs,
                                                            config.iter_per_epoch * config.n_epochs))
    logger.warning("running time: " + datetime.datetime.now().strftime(' [%H:%M] %d/%m/%y'))   
    logger.warning("supervised sample: {}, unsupervised sample: {}".format(config.labeled_num,
                                                                           config.unlabeled_num))
    logger.critical("architecture: {}, backbone: {}".format(args.architecture,
                                                            "nothing" if args.backbone is None
                                                            else args.backbone))
    tensorboard = Tensorboard(config=config)
    trainer = Trainer(config, train_loader=train_loader,
                      valid_set=val_dataset, logger=logger,
                      my_wandb=tensorboard)
    trainer.run()
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Medical Semi-supervised Semantic Segmentation')
    # network architectures
    parser.add_argument("-a", "--architecture", default='vnet', type=str,
                        help="select the architecture in use")
    parser.add_argument("-b", "--backbone", default=None, type=str,
                        help="select the architecture in use")
    # experimental settings
    parser.add_argument("--unsup_weight", default=1.0, type=float,
                        help="unsupervised weight for the semi-supervised learning")
    parser.add_argument("--labeled_num", default=7, type=int,
                        help="number of the labelled sample")
    parser.add_argument("--max_iterations", default=50000, type=int,
                        help="max iterations for the training")
    # pvc
    # parser.add_argument("--pvc", action='store_true', help="use pvc or not")

    # wandb
    parser.add_argument("--api_key", help="wandb key for docker")

    cmd_line_var = parser.parse_args()

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    # random.seed(config.seed)
    # numpy.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)

    main(cmd_line_var)