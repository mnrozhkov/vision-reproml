
# Use notebooks to get and preprocess data and train model

### catsanddogs.ipynb

* get data
* preprocess data


### datapreprocessor.ipynb

* train model
* save checkpoint


# Build docker image

1. create file config/dev/.env with content:

    * COMPOSE_FILE - нужная версия docker-compose для dev среды
    * PYTORCH_MODELS_FOLDER - путь к папке с моделями pytorch

    пример: 

        COMPOSE_FILE=docker-compose.yml:config/dev/docker-compose.dev.yml
        PYTORCH_MODELS_FOLDER=/home/arnolod/models

2. run build:

    ln -sf config/dev/.env && docker-compose build
    
# Run

    docker-compose run -p 8888:8888 vision-reproml
