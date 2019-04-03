
# Use notebooks to get and preprocess data and train model

### catsanddogs.ipynb

* get data
* preprocess data


### datapreprocessor.ipynb

* train model
* save checkpoint


# Download pretrained models

resnet50

    https://download.pytorch.org/models/resnet50-19c8e357.pth


# Build docker image

1. create file config/dev/.env with content:

    * COMPOSE_FILE - нужная версия docker-compose для dev среды

    пример: 

        COMPOSE_FILE=docker-compose.yml:config/dev/docker-compose.dev.yml

2. run build:

    ln -sf config/dev/.env && docker-compose build
    
# Run
    
    export PYTORCH_MODELS_FOLDER=/path/to/folder/with/pytorch/models
    docker-compose run -p 8888:8888 -v $PYTORCH_MODELS_FOLDER:/home/user/.torch/models vision-reproml
