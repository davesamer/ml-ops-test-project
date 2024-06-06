#!/bin/bash

#source ../../config.env

# install docker and docker compose
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce

# start mlflow 
sudo systemctl start docker
DOCKERIMAGE="davesamer/churn_prediction_model_online:latest"
sudo docker pull "${DOCKERIMAGE}"
sudo docker run -p 8000:8080 "${DOCKERIMAGE}"
