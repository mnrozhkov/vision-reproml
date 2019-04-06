
import argparse
from collections import OrderedDict
import json
import logging
from mlflow import log_artifact, log_metric, log_param
import os
import torch
from torchvision import datasets
import yaml

from src.evaluate.evaluate import evaluate
from src.transforms.transforms import data_transforms


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    log_artifact(args.config)

    config = yaml.load(open(args.config))

    torch.manual_seed(config['random_seed'])

    model_conv = torch.load(os.path.join(config['models_folder'], config['model_name']))

    image_dataset = datasets.ImageFolder(os.path.join(config['dataset_dir'], 'test'), data_transforms['test'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=config['batch_size'], shuffle=True)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_dict = evaluate(model_conv, dataloader, device)

    log_param('batch_size', config['batch_size'])

    # prepare test metrics and report dicts
    precision = eval_dict['precision']
    recall = eval_dict['recall']
    f1 = eval_dict['f1']
    roc_auc = eval_dict['roc_auc']
    cm = eval_dict['cm']

    log_metric('precision', precision)
    log_metric('recall', recall)
    log_metric('f1', f1)
    log_metric('roc_auc', roc_auc)

    log_metric('true_negatives', cm[0][0])
    log_metric('false_negatives', cm[1][0])
    log_metric('true_positives', cm[0][1])
    log_metric('false_positives', cm[1][1])

    test_metrics = OrderedDict([
        ('precision', precision),
        ('recall', recall),
        ('f1', f1),
        ('roc_auc', roc_auc),
        ('confusion_matrix', cm.tolist())
    ])
    test_report = OrderedDict([
        ('model_name', config['model_name']),
        ('metrics', test_metrics)
    ])

    # save test evaluation report
    evaluation_report_path = os.path.join(
        config['reports_folder'],
        '{}_evaluate_report.json'.format(config['model_name'])
    )
    logging.debug("evaluation_report_path: " + evaluation_report_path)
    with open(evaluation_report_path, 'w') as save_evaluation_results_file:
        json.dump(
            obj=test_report,
            fp=save_evaluation_results_file,
            ensure_ascii=False,
            indent=4
        )

    log_artifact(evaluation_report_path)

    # save test evaluation metrics
    test_metrics_path = os.path.join(config['reports_folder'], 'test_metrics.json')
    logging.debug("test_metrics_path: " + test_metrics_path)
    with open(test_metrics_path, 'w') as test_metrics_file:
        json.dump(
            obj=test_metrics,
            fp=test_metrics_file,
            ensure_ascii=False,
            indent=4
        )

    log_artifact(test_metrics_path)

