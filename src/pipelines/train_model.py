
import argparse
from collections import OrderedDict
import json
import logging
from mlflow import log_artifact, log_metric, log_param
import os
import torch
from torch import optim, nn
from torchvision import datasets, models
import yaml

from src.train.train_model import train_model
from src.transforms.transforms import data_transforms


logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    log_artifact(args.config)

    config = yaml.load(open(args.config))

    torch.manual_seed(config['random_seed'])

    image_datasets = {x: datasets.ImageFolder(os.path.join(config['dataset_dir'], x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_conv = models.resnet50(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=config['step_size'], gamma=config['gamma'])

    model_conv, best_val_loss, best_val_acc = train_model(model_conv,
                                                          criterion,
                                                          optimizer_conv,
                                                          exp_lr_scheduler,
                                                          dataloaders,
                                                          dataset_sizes,
                                                          device,
                                                          num_epochs=config['epochs'])

    logging.info('{} image folder'.format(config['dataset_dir']))
    log_param('train_size', dataset_sizes['train'])
    log_param('valid_size', dataset_sizes['val'])
    log_param('batch_size', config['batch_size'])

    model_full_name = os.path.join(config['models_folder'], config['model_name'])
    torch.save(model_conv, model_full_name)

    log_artifact(model_full_name)

    log_metric('best_val_loss', best_val_loss)
    log_metric('best_val_acc', best_val_acc.tolist())

    train_report_filename = os.path.join(
                config['reports_folder'],
                '{}_train_report.json'.format(config['model_name'])
            )
    with open(
            train_report_filename,
            'w'
    ) as train_report_file:

        report_dict = OrderedDict([

            ('train_size', dataset_sizes['train']),
            ('valid_size', dataset_sizes['val']),
            ('epochs', config['epochs']),
            ('best_val_loss', best_val_loss),
            ('best_val_acc', best_val_acc.tolist())
        ])

        json.dump(
            obj=report_dict,
            fp=train_report_file,
            ensure_ascii=False,
            indent=4
        )

    log_artifact(train_report_filename)
