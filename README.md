
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

## Python scripts

pipelines scripts (python) location: src/pipelines

pipelines:

* __prepare_configs.py__: load config/pipeline_config.yml and split it into configs specific for next stages

* __prepare_dataset.py__: unzip dataset archive, split source dataset into train/val/test sets

* __train_model.py__: train model and log training params, artifacts and metrics to mlflow 

* __evaluate_model.py__: evaluate model and log evaluating params, artifacts and metrics to mlflow


## Dvc pipelines

* pipeline_prepare_configs.dvc

    
    dvc run -f pipeline_prepare_configs.dvc \
        -d src/pipelines/prepare_configs.py \
        -d config/pipeline_config.yml \
        -o experiments/prepare_dataset_config.yml \
        -o experiments/train_model_config.yml \
        -o experiments/evaluate_model_config.yml \
        python src/pipelines/prepare_configs.py --config=config/pipeline_config.yml

this pipeline:

1) creates configs `prepare_dataset_config.yml`, `train_model_config.yml`, `evaluate_model_config.yml`
2) generate stage file `pipeline_prepare_configs.dvc`

To reproduce: `dvc repro pipeline_prepare_configs.dvc`

        
* pipeline_prepare_dataset.dvc

    
    dvc run -f pipeline_prepare_dataset.dvc \
        -d src/pipelines/prepare_dataset.py \
        -d experiments/prepare_dataset_config.yml \
        -O data/train \
        -O data/val \
        -O data/test \
        -O data/sample \
        python src/pipelines/prepare_dataset.py --config=experiments/prepare_dataset_config.yml

this pipeline:

1) unzip dataset archive, split dataset on train/val/test sets and create sample (short version) dataset
2) generate stage file `pipeline_prepare_dataset.dvc`        

To reproduce: `dvc repro pipeline_prepare_dataset.dvc`


* pipeline_train_model.dvc

    
    dvc run -f pipeline_train_model.dvc \
        -d src/pipelines/train_model.py \
        -d experiments/train_model_config.yml \
        -d data/train \
        -d data/val \
        -d data/test \
        -d data/sample \
        -o models/model.pth \
        -o experiments/model.pth_train_report.json \
        python src/pipelines/train_model.py --config=experiments/train_model_config.yml

this pipeline:

1) trains and save model
2) save training report
3) generate stage file `pipeline_train_model.dvc`        

To reproduce: `dvc repro pipeline_train_model.dvc`


* pipeline_evaluate_model.dvc


    dvc run -f pipeline_evaluate_model.dvc \
        -d src/pipelines/evaluate_model.py \
        -d experiments/evaluate_model_config.yml \
        -d models/model.pth \
        -o experiments/model.pth_evaluate_report.json \
        python src/pipelines/evaluate_model.py --config=experiments/evaluate_model_config.yml
        

this pipeline:

1) evaluate model
2) save evaluating report
3) generate stage file `pipeline_evaluate_model.dvc`

To reproduce: `dvc repro pipeline_evaluate_model.dvc`
