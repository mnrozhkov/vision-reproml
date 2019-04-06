
import argparse
import os
import yaml


def split_common_config():

    """
    Split common config into configs for steps
    """

    # add arguments
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--config', dest='config', required=True)

    args = args_parser.parse_args()

    # read config and get config sections

    config = yaml.load(open(args.config))
    dataset_config = config['dataset']
    model_config = config['model']
    train_config = config['train']
    evaluate_config = config['evaluate']
    report_config = config['report']
    split_config = config['split_config']

    # create folder for script configs if not exists
    os.makedirs(split_config['folder'], exist_ok=True)

    prepare_dataset_config = dict(
        data_dir=dataset_config['data_dir'],
        dataset_archive_name=dataset_config['dataset_archive_name'],
        sample=dataset_config['sample']
    )

    with open('{}/{}'.format(split_config['folder'], 'prepare_dataset_config.yml'), 'w') as prepare_dataset_yml:
        yaml.dump(
            data=prepare_dataset_config,
            stream=prepare_dataset_yml,
            default_flow_style=False
        )

    # train model config
    train_model_config = dict(
        dataset_dir=dataset_config['dataset_dir'],
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        models_folder=model_config['models_folder'],
        model_name=model_config['model_name'],
        reports_folder=report_config['reports_folder'],
        lr=model_config['lr'],
        momentum=model_config['momentum'],
        step_size=model_config['step_size'],
        gamma=model_config['gamma'],
        random_seed=config['random_seed']
    )

    with open('{}/{}'.format(split_config['folder'], 'train_model_config.yml'), 'w') as train_model_yml:
        yaml.dump(
            data=train_model_config,
            stream=train_model_yml,
            default_flow_style=False
        )

    # evaluate model config
    evaluate_model_config = dict(
        dataset_dir=dataset_config['dataset_dir'],
        models_folder=model_config['models_folder'],
        model_name=model_config['model_name'],
        batch_size=evaluate_config['batch_size'],
        reports_folder=report_config['reports_folder'],
        random_seed=config['random_seed']
    )

    with open('{}/{}'.format(split_config['folder'], 'evaluate_model_config.yml'), 'w') as evaluate_model_yml:
        yaml.dump(
            data=evaluate_model_config,
            stream=evaluate_model_yml,
            default_flow_style=False
        )


if __name__ == '__main__':

    split_common_config()
