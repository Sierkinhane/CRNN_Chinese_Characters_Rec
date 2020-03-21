import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets

from tensorboardX import SummaryWriter


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss(reduction='sum')

    optimizer = utils.get_optimizer(config, model)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()

if __name__ == '__main__':

    main()