
# Download dataset archive

#### download archive to folder _data_

   ###### [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)
    

# Use notebook catsanddogs.ipynb to run pipelines

    prepare configs -> prepare dataset -> train model -> evaluate model


# Download pretrained models

1. create directory for torchvision pretrained models (outside of repository)

2. download in this directory resnet50


    https://download.pytorch.org/models/resnet50-19c8e357.pth


# Build docker image

    docker-compose build
    
# Run
    
    export PYTORCH_MODELS_FOLDER=/path/to/folder/with/pytorch/models
    docker-compose run -p 8888:8888 -p 1234:1234 -v $PYTORCH_MODELS_FOLDER:/home/user/.torch/models vision-reproml


# Run mlflow ui

    mlflow ui --host=0.0.0.0 --port=1234
    
    
# Pipelines description

pipelines scripts (python) location: src/pipelines

pipelines:

* __prepare_configs.py__: load config/pipeline_config.yml and split it into configs specific for next stages

* __prepare_dataset.py__: unzip dataset archive, split source dataset into train/val/test sets

* __train_model.py__: train model and log training params, artifacts and metrics to mlflow 

* __evaluate_model.py__: evaluate model and log evaluating params, artifacts and metrics to mlflow