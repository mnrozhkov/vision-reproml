import argparse
import os
import re
import shutil
import yaml

from src.utils import unzip, cp_n_files


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    config = yaml.load(open(args.config))

    # unzip archives
    print('unzip archive...')
    unzip(os.path.join(config['data_dir'], config['dataset_archive_name']), config['data_dir'])
    unzip(os.path.join(config['data_dir'], 'train.zip'), config['data_dir'])

    # create directories
    print('\n\ncreate directories...')
    train_dir = os.path.join(config['data_dir'], 'train')

    train_dogs_dir = '{train_dir}/dogs'.format(train_dir=train_dir)
    train_cats_dir = '{train_dir}/cats'.format(train_dir=train_dir)

    os.makedirs(train_dogs_dir, exist_ok=True)
    os.makedirs(train_cats_dir, exist_ok=True)

    val_dir = os.path.join(config['data_dir'], 'val')

    val_dogs_dir = '{val_dir}/dogs'.format(val_dir=val_dir)
    val_cats_dir = '{val_dir}/cats'.format(val_dir=val_dir)

    os.makedirs(val_dogs_dir, exist_ok=True)
    os.makedirs(val_cats_dir, exist_ok=True)

    test_dir = os.path.join(config['data_dir'], 'test')
    test_dogs_dir = '{test_dir}/dogs'.format(test_dir=test_dir)
    test_cats_dir = '{test_dir}/cats'.format(test_dir=test_dir)

    os.makedirs(test_dogs_dir, exist_ok=True)
    os.makedirs(test_cats_dir, exist_ok=True)

    # Split train images by classes
    print('\n\nsplit train images by classes...')
    files = os.listdir(train_dir)

    # Move all train cat images to cats folder, dog images to dogs folder
    for f in files:
        catSearchObj = re.search("cat", f)
        dogSearchObj = re.search("dog", f)
        if catSearchObj:
            shutil.move('{train_dir}/{f}'.format(train_dir=train_dir, f=f), train_cats_dir)
        elif dogSearchObj:
            shutil.move('{train_dir}/{f}'.format(train_dir=train_dir, f=f), train_dogs_dir)

    # Create validation dataset - extract each image with number started from '5' from trainset
    print('\n\ncreate validation dataset...')
    files = os.listdir(train_dogs_dir)

    for f in files:
        validationDogsSearchObj = re.search("5\d\d\d", f)
        if validationDogsSearchObj:
            shutil.move('{train_dogs_dir}/{f}'.format(train_dogs_dir=train_dogs_dir, f=f), val_dogs_dir)

    files = os.listdir(train_cats_dir)

    for f in files:
        validationCatsSearchObj = re.search("5\d\d\d", f)
        if validationCatsSearchObj:
            shutil.move('{train_cats_dir}/{f}'.format(train_cats_dir=train_cats_dir, f=f), val_cats_dir)

    # Create test dataset - extract each image with number started from '4' from trainset
    print('\n\ncreate test dataset...')
    files = os.listdir(train_dogs_dir)

    for f in files:
        validationDogsSearchObj = re.search("[4]\d\d\d", f)
        if validationDogsSearchObj:
            shutil.move('{train_dogs_dir}/{f}'.format(train_dogs_dir=train_dogs_dir, f=f), test_dogs_dir)

    files = os.listdir(train_cats_dir)

    for f in files:
        validationDogsSearchObj = re.search("[4]\d\d\d", f)
        if validationDogsSearchObj:
            shutil.move('{train_cats_dir}/{f}'.format(train_cats_dir=train_cats_dir, f=f), test_cats_dir)

    # Create sample dataset (short version of dataset)
    print('\n\ncreate sample dataset...')
    sample_config = config['sample']

    sample_dir = sample_config['folder']
    sample_dir_train = '{}/train'.format(sample_dir)
    sample_dir_train_cats = '{}/cats'.format(sample_dir_train)
    sample_dir_train_dogs = '{}/dogs'.format(sample_dir_train)

    sample_dir_val = '{}/val'.format(sample_dir)
    sample_dir_val_cats = '{}/cats'.format(sample_dir_val)
    sample_dir_val_dogs = '{}/dogs'.format(sample_dir_val)

    sample_dir_test = '{}/test'.format(sample_dir)
    sample_dir_test_cats = '{}/cats'.format(sample_dir_test)
    sample_dir_test_dogs = '{}/dogs'.format(sample_dir_test)

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(sample_dir_train_cats, exist_ok=True)
    os.makedirs(sample_dir_train_dogs, exist_ok=True)
    os.makedirs(sample_dir_val_cats, exist_ok=True)
    os.makedirs(sample_dir_val_dogs, exist_ok=True)
    os.makedirs(sample_dir_test_cats, exist_ok=True)
    os.makedirs(sample_dir_test_dogs, exist_ok=True)

    cp_n_files(train_cats_dir, sample_dir_train_cats, sample_config['train_size'])
    cp_n_files(train_dogs_dir, sample_dir_train_dogs, sample_config['train_size'])

    cp_n_files(val_cats_dir, sample_dir_val_cats, sample_config['val_size'])
    cp_n_files(val_dogs_dir, sample_dir_val_dogs, sample_config['val_size'])

    cp_n_files(test_cats_dir, sample_dir_test_cats, sample_config['test_size'])
    cp_n_files(test_dogs_dir, sample_dir_test_dogs, sample_config['test_size'])

