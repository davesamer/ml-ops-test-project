

# Set Up

## Setting up MLFlow 

1) Run ```sudo docker compose --env-file config.env up -d --build```
2) Access minio under localhost:9011
3) Create access key
4) Create bucket called mlflow
5) docker compose down
6) Update config.env file
7) Run ```sudo docker compose --env-file config.env up -d --build```


## MLOps Workflow

### Model Lifecycle

- When model training is finished, register model
- Model is loaded in validation. If model successfully passes all checks asign "Challenger" alias. Add tag model_validation_status” and set the value to “PENDING” as the tests execute, and then update it to “PASSED” or “FAILED” when the pipeline is complete.
- Current production model has alias "Champion". In Deployment pipeline "Challenger" model is compared to "Champion" model. (Either offline on held-out set or online, e.g. A/B testing). If currently no ML model is in production compare against baseline 
- After comparison - if challenger is better than champion - set chllenger to champion and deploy to production


### Overview Scripts
- feature_retrieval.py: accessing and preprocessing the data needed for model training & predictions
- feature_validation.py: running tests to ensure that input data is valid
- model_training.py: training & evaluating a model based on input data and logging experiment with mlflow
- model_validation.py: running tests to ensure validity/compliance with company policies of the model
- model_handover.py: registrating model to mlflow model registry

### Run scripts
- feature_retrieval ```python feature_retrieval.py```
- feature_validation ```pytest feature_validation.py```(must be executed from code/ directory) 


### CI/CD Pipelines:
- training_pipeline.yaml: 
    - Task: Automatizes process of model training. Runs script above sequentially. If one step fails, pipeline stops. 
    - Trigger: Pipeline is triggered by pushing into main or manually
- batch_deployment_pipeline.yaml: 
    - Task: Automatizes process of deploying ML model for batch prediction to Azure using MLFlow Deployment
    - Trigger: Manually
- online_deployment_pipeline.yaml:
    - Task: Automatizes process of deploying ML model for online prediction to Azure using MLFlow Deployment
    - Trigger: Manually

### Git Workflow:
- master: anything in the master branch is deployable
- branches are used to develop new features
- to merge branches into master pull request must be made which triggers code review. 
- before code review code is automatically tested; pull request fails if tests fail. 

# TODO
- Next goal: Serving "champion model" on azure infrastructure
    - IaC Azure VM for model deployment & Blob Storage for data
    - Test deployment with mlflow locally / build docker image
    - Pipeline for model deployment (batch or online)
        - model comparison --> using best performing model and set it to champion (offline on hold out dataset)

- write unit/integration tests and integrate into pipelines

# Infrastructure
- Azure Blob Storage (to store training data)
- Azure Container Registry
- Resource Group
- VPN & Subnetwork
- IP Address
- Security Groups
- Network Interface & Network Interface Security Group association
- VM (with cloud init file to run docker image)

# TODO
- 