# Code snippets
This file just holds all the code snippets from the learning paths, and can serve as a good review of the Python SDK.
The partial URL is shown to give the snippets some context.

make-data-available-azure-machine-learning_3-create-datastore.html
```python
blob_datastore = AzureBlobDatastore(
    			name = "blob_example",
    			description = "Datastore pointing to a blob container",
    			account_name = "mytestblobstore",
    			container_name = "data-container",
    			credentials = AccountKeyCredentials(
        			account_key="XXXxxxXXXxXXXXxxXXX"
    			),
)
ml_client.create_or_update(blob_datastore)

```
make-data-available-azure-machine-learning_3-create-datastore.html
```python
blob_datastore = AzureBlobDatastore(
name="blob_sas_example",
description="Datastore pointing to a blob container”,
account_name="mytestblobstore",
container_name="data-container",
credentials=SasTokenCredentials(
sas_token="?xx=XXXX-XX-XX&xx=xxxx&xxx=xxx&xx=xxxxxxxxxxx&xx=XXXX-XX-XXXXX:XX:XXX&xx=XXXX-XX-XXXXX:XX:XXX&xxx=xxxxx&xxx=XXxXXXxxxxxXXXXXXXxXxxxXXXXXxxXXXXXxXXXXxXXXxXXxXX"
),
)
ml_client.create_or_update(blob_datastore)

```
automate-machine-learning-workflows_4-github-actions.html
```yaml
name: Train model

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: install az ml extension
      run: az extension add -n ml -y
    - name: azure login
      uses: azure/login@v1
      with:
        creds: ${{secrets.AZURE_CREDENTIALS}}
    - name: set current directory
      run: cd src
    - name: run pipeline
      run: az ml job create --file src/aml_service/pipeline-job.yml --resource-group dev-ml-rg --workspace-name dev-ml-ws

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="<description>",
    name="<name>",
    version="<version>"
)

ml_client.data.create_or_update(my_data)

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head(10))

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="<description>",
    name="<name>",
    version='<version>'
)

ml_client.data.create_or_update(my_data)

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

data_path = args.input_data
all_files = glob.glob(data_path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```yaml
type: mltable

paths:
  - pattern: ./*.txt
transformations:
  - read_delimited:
      delimiter: ','
      encoding: ascii
      header: all_files_same_headers

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<path-including-mltable-file>'

my_data = Data(
    path=my_path,
    type=AssetTypes.MLTABLE,
    description="<description>",
    name="<name>",
    version='<version>'
)

ml_client.data.create_or_update(my_data)

```
make-data-available-azure-machine-learning_4-create-data-asset.html
```python
import argparse
import mltable
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

tbl = mltable.load(args.input_data)
df = tbl.to_pandas_dataframe()

print(df.head(10))

```
automate-machine-learning-workflows_2-machine-learning-pipelines.html
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: nyc-taxi-pipeline-example
experiment_name: nyc-taxi-pipeline-example
jobs:

  transform-job:
    type: command
      raw_data:
          type: uri_folder
          path: ./data
    outputs:
      transformed_data:
        mode: rw_mount
    code: src/transform
    environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
    compute: azureml:cpu-cluster
    command: >-
      python transform.py
      --raw_data ${{inputs.raw_data}}
      --transformed_data ${{outputs.transformed_data}}

  train-job:
    type: command
    inputs:
      training_data: ${{parent.jobs.transform-job.outputs.transformed_data}}
    outputs:
      model_output:
        mode: rw_mount
      test_data:
        mode: rw_mount
    code: src/train
    environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
    compute: azureml:cpu-cluster
    command: >-
      python train.py
      --training_data ${{inputs.training_data}}
      --test_data ${{outputs.test_data}}
      --model_output ${{outputs.model_output}}

```
automate-machine-learning-workflows_2-machine-learning-pipelines.html
```bash
az ml job create --file pipeline-job.yml

```
continuous-deployment-for-machine-learning_2-set-up-environments-for-development-production.html
```yaml
trigger:
- main

stages:
- stage: deployDev
  displayName: 'Deploy to development environment'
  jobs:
    - deployment: publishPipeline
      displayName: 'Model Training'
      pool:
        vmImage: 'Ubuntu-18.04'
      environment: dev
      strategy:
       runOnce:
         deploy:
          steps:
          - template: aml-steps.yml
            parameters:
              serviceconnectionname: 'spn-aml-workspace-dev'

```
continuous-deployment-for-machine-learning_2-set-up-environments-for-development-production.html
```yaml
name: Train model

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    environment:
        name: dev
    steps:
    - name: check out repo
      uses: actions/checkout@v2
    - name: install az ml extension
      run: az extension add -n ml -y
    - name: azure login
      uses: azure/login@v1
      with:
        creds: ${{secrets.AZURE_CREDENTIALS}}
    - name: set current directory
      run: cd src
    - name: run pipeline
      run: az ml job create --file src/aml_service/pipeline-job.yml --resource-group dev-ml-rg --workspace-name dev-ml-ws

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```python
import json
import joblib
import numpy as np
import os

# called when the deployment is created or updated
def init():
    global model
    # get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    # get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # get a prediction from the model
    predictions = model.predict(data)
    # return the predictions as any JSON serializable format
    return predictions.tolist()

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```yaml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda-env.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```python
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

model = Model(path="./model",

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    environment="deployment-environment",
    code_configuration=CodeConfiguration(
        code="./src", scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```python
# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()

```
deploy-model-managed-online-endpoint_4-eploy-custom-model-managed-online-endpoint.html
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")

```
run-pipelines-azure-machine-learning_3-create-pipeline.html
```python
from azure.ai.ml.dsl import pipeline

@pipeline()
def pipeline_function_name(pipeline_job_input):
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }

```
run-pipelines-azure-machine-learning_3-create-pipeline.html
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE,
    path="azureml:data:1"
))

```
run-pipelines-azure-machine-learning_3-create-pipeline.html
```python
print(pipeline_job)

```
run-pipelines-azure-machine-learning_3-create-pipeline.html
```yaml
display_name: pipeline_function_name
type: pipeline
inputs:
  pipeline_job_input:
    type: uri_file
    path: azureml:data:1
outputs:
  pipeline_job_transformed_data: null
  pipeline_job_trained_model: null
jobs:
  prep_data:
    type: command
    inputs:
      input_data:
        path: ${{parent.inputs.pipeline_job_input}}
    outputs:
      output_data: ${{parent.outputs.pipeline_job_transformed_data}}
tags: {}
properties: {}
settings: {}

```
deploy-model-batch-endpoint_3-deploy-your-mlflow-model-batch-endpoint.html
```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_name = 'mlflow-model'
model = ml_client.models.create_or_update(
    Model(name=model_name, path='./model', type=AssetTypes.MLFLOW_MODEL)
)

```
deploy-model-batch-endpoint_3-deploy-your-mlflow-model-batch-endpoint.html
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size=2,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)

```
run-training-script-command-job-azure-machine-learning_3-run-script-command-job.html
```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )

```
run-training-script-command-job-azure-machine-learning_3-run-script-command-job.html
```python
# submit job
returned_job = ml_client.create_or_update(job)

```
explore-azure-machine-learning-workspace-resources-assets_2-provision.html
```python
from azure.ai.ml.entities import Workspace

workspace_name = "mlw-example"

ws_basic = Workspace(
    name=workspace_name,
    location="eastus",
    display_name="Basic workspace-example",
    description="This example shows how to create a basic workspace",
)
ml_client.workspaces.begin_create(ws_basic)

```
deploy-model-managed-online-endpoint_2-explore-managed-online-endpoints.html
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="endpoint-example",
    description="Online endpoint",
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()

```
explore-developer-tools-for-workspace-interaction_4-explore-cli.html
```bash
az extension add -n ml -y

```
explore-developer-tools-for-workspace-interaction_4-explore-cli.html
```bash
az ml -h

```
explore-developer-tools-for-workspace-interaction_4-explore-cli.html
```bash
az ml compute create --name aml-cluster --size STANDARD_DS3_v2 --min-instances 0 --max-instances 5 --type AmlCompute --resource-group my-resource-group --workspace-name my-workspace

```
explore-developer-tools-for-workspace-interaction_4-explore-cli.html
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json
name: aml-cluster
type: amlcompute
size: STANDARD_DS3_v2
min_instances: 0
max_instances: 5

```
explore-developer-tools-for-workspace-interaction_4-explore-cli.html
```bash
az ml compute create --file compute.yml --resource-group my-resource-group --workspace-name my-workspace

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
# set pipeline level compute
pipeline_job.settings.default_compute = "aml-cluster"

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
print(pipeline_job)

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute",
    interval=1,
)

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()

```
run-pipelines-azure-machine-learning_4-run-pipeline-job.html
```python
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_4-configure-early-termination.html
```python
from azure.ai.ml.sweep import BanditPolicy

sweep_job.early_termination = BanditPolicy(
    slack_amount = 0.2,
    delay_evaluation = 5,
    evaluation_interval = 1
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_4-configure-early-termination.html
```python
from azure.ai.ml.sweep import MedianStoppingPolicy

sweep_job.early_termination = MedianStoppingPolicy(
    delay_evaluation = 5,
    evaluation_interval = 1
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_4-configure-early-termination.html
```python
from azure.ai.ml.sweep import TruncationSelectionPolicy

sweep_job.early_termination = TruncationSelectionPolicy(
    evaluation_interval=1,
    truncation_percentage=20,
    delay_evaluation=4
)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
experiments = mlflow.list_experiments(max_results=2)
for exp in experiments:
    print(exp.name)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
from mlflow.entities import ViewType

experiments = mlflow.list_experiments(view_type=ViewType.ALL)
for exp in experiments:
    print(exp.name)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
mlflow.search_runs(exp.experiment_id)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=2)

```
train-models-training-mlflow-jobs_3-view-metrics-evaluate-models.html
```python
mlflow.search_runs(
    exp.experiment_id, filter_string="params.num_boost_round='100'", max_results=2
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_2-define-search-space.html
```python
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[0.01, 0.1, 1]),
    learning_rate=Normal(mu=10, sigma=3),
)

```
introduction-development-operations-principles-for-machine-learn_4-integrate-azure-development-operations-tools.html
```bash
az ad sp create-for-rbac --name "github-aml-sp" --role contributor \
                            --scopes /subscriptions/<subscription-id>/resourceGroups/<group-name>/providers/Microsoft.MachineLearningServices/workspaces/<workspace-name> \
                            --sdk-auth

```
run-training-script-command-job-azure-machine-learning_4-use-parameters-command-job.html
```python
# import libraries
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)

```
run-training-script-command-job-azure-machine-learning_4-use-parameters-command-job.html
```bash
python train.py --training_data diabetes.csv

```
run-training-script-command-job-azure-machine-learning_4-use-parameters-command-job.html
```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )

```
deploy-model-batch-endpoint_4-deploy-custom-model-batch-endpoint.html
```python
import os
import mlflow
import pandas as pd


def init():
    global model

    # get the path to the registered model file and load it
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.pyfunc.load(model_path)


def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        pred = model.predict(data)

        df = pd.DataFrame(pred, columns=["predictions"])
        df["file"] = os.path.basename(file_path)
        resultList.extend(df.values)

    return resultList

```
deploy-model-batch-endpoint_4-deploy-custom-model-batch-endpoint.html
```yaml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pandas
  - pip
  - pip:
      - azureml-core
      - mlflow

```
deploy-model-batch-endpoint_4-deploy-custom-model-batch-endpoint.html
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda-env.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)

```
deploy-model-batch-endpoint_4-deploy-custom-model-batch-endpoint.html
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    code_path="./code",
    scoring_script="score.py",
    environment=env,
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size=2,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)

```
automate-machine-learning-workflows_3-azure-pipelines.html
```yaml
trigger:
- main

stages:
- stage: deployDev
  displayName: 'Deploy to development environment'
  jobs:
    - deployment: publishPipeline
      displayName: 'Model Training'
      pool:
        vmImage: 'Ubuntu-18.04'
      environment: dev
      strategy:
       runOnce:
         deploy:
          steps:
          - template: aml-steps.yml
            parameters:
              serviceconnectionname: 'spn-aml-workspace-dev'

```
automate-machine-learning-workflows_3-azure-pipelines.html
```bash
parameters:
- name: serviceconnectionname
  default: ''

steps:
- checkout: self

- script: az extension add -n ml -y
  displayName: 'Install Azure ML CLI v2'

- task: AzureCLI@2
  inputs:
    azureSubscription: ${{ parameters.serviceconnectionname }}
    scriptType: bash
    scriptLocation: inlineScript
    workingDirectory: $(Build.SourcesDirectory)
    inlineScript: |
      cd src
      az ml job create --file aml_service/pipeline-job.yml --resource-group dev-ml-rg --workspace-name dev-ml-ws
  displayName: 'Run Azure Machine Learning Pipeline'

```
deploy-model-batch-endpoint_2-explore-batch-endpoints.html
```python
# create a batch endpoint
endpoint = BatchEndpoint(
    name="endpoint-example",
    description="A batch endpoint",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint)

```
deploy-model-batch-endpoint_2-explore-batch-endpoints.html
```python
from azure.ai.ml.entities import AmlCompute

cpu_cluster = AmlCompute(
    name="aml-cluster",
    type="amlcompute",
    size="STANDARD_DS11_V2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
    tier="Dedicated",
)

cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)

```
deploy-model-batch-endpoint_5-monitor-batch-endpoints.html
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

input = Input(type=AssetTypes.URI_FOLDER, path="azureml:new-data:1")

job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint.name,
    input=input)

```
find-best-classification-model-automated-machine-learning_3-run-job.html
```python
from azure.ai.ml import automl

# configure the classification job
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)

```
find-best-classification-model-automated-machine-learning_3-run-job.html
```python
from azure.ai.ml.automl import ClassificationPrimaryMetrics

list(ClassificationPrimaryMetrics)

```
find-best-classification-model-automated-machine-learning_3-run-job.html
```python
classification_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=5,
    enable_early_termination=True,
)

```
find-best-classification-model-automated-machine-learning_3-run-job.html
```python
# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)

```
find-best-classification-model-automated-machine-learning_3-run-job.html
```python
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)

```
run-training-script-command-job-azure-machine-learning_2-convert-notebook-script.html
```python
# read and visualize the data
print("Reading data...")
df = pd.read_csv('diabetes.csv')
df.head()

# split data
print("Splitting data...")
X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

```
run-training-script-command-job-azure-machine-learning_2-convert-notebook-script.html
```python
def main(csv_file):
    # read data
    df = get_data(csv_file)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    return df

# function that splits the data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

```
run-training-script-command-job-azure-machine-learning_2-convert-notebook-script.html
```bash
python train.py

```
run-pipelines-azure-machine-learning_2-create-components.html
```python
# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# setup arg parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("--input_data", dest='input_data',
                    type=str)
parser.add_argument("--output_data", dest='output_data',
                    type=str)

# parse args
args = parser.parse_args()

# read the data
df = pd.read_csv(args.input_data)

# remove missing values
df = df.dropna()

# normalize the data
scaler = MinMaxScaler()
num_cols = ['feature1','feature2','feature3','feature4']
df[num_cols] = scaler.fit_transform(df[num_cols])

# save the data as a csv
output_df = df.to_csv(
    (Path(args.output_data) / "prepped-data.csv"),
    index = False
)

```
run-pipelines-azure-machine-learning_2-create-components.html
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_data
display_name: Prepare training data
version: 1
type: command
inputs:
  input_data:
    type: uri_file
outputs:
  output_data:
    type: uri_file
code: ./src
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
command: >-
  python prep.py
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}

```
run-pipelines-azure-machine-learning_2-create-components.html
```python
from azure.ai.ml import load_component
parent_dir = ""

loaded_component_prep = load_component(source=parent_dir + "./prep.yml")

```
run-pipelines-azure-machine-learning_2-create-components.html
```python
prep = ml_client.components.create_or_update(prepare_data_component)

```
deploy-model-managed-online-endpoint_5-monitor-online-endpoints.html
```bash
{
  "data":[
      [0.1,2.3,4.1,2.0], // 1st case
      [0.2,1.8,3.9,2.1],  // 2nd case,
      ...
  ]
}

```
deploy-model-managed-online-endpoint_5-monitor-online-endpoints.html
```python
# test the blue deployment with some sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="sample-data.json",
)

if response[1]=='1':
    print("Yes")
else:
    print ("No")

```
explore-developer-tools-for-workspace-interaction_3-explore-python-sdk.html
```bash
pip install azure-ai-ml

```
explore-developer-tools-for-workspace-interaction_3-explore-python-sdk.html
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

```
explore-developer-tools-for-workspace-interaction_3-explore-python-sdk.html
```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    experiment_name="train-model"
)

# connect to workspace and submit job
returned_job = ml_client.create_or_update(job)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_5-use-sweep-job-hyperparameter-tuning.html
```python
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("data.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_5-use-sweep-job-hyperparameter-tuning.html
```python
from azure.ai.ml import command

# configure command job as base
job = command(
    code="./src",
    command="python train.py --regularization ${{inputs.regularization}}",
    inputs={
        "reg_rate": 0.01,
    },
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    )

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_5-use-sweep-job-hyperparameter-tuning.html
```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = job(
    reg_rate=Choice(values=[0.01, 0.1, 1]),
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_5-use-sweep-job-hyperparameter-tuning.html
```python
from azure.ai.ml import MLClient

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="Accuracy",
    goal="Maximize",
)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-example"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=4, max_concurrent_trials=2, timeout=7200)

# submit the sweep
returned_sweep_job = ml_client.create_or_update(sweep_job)

```
source-control-for-machine-learning-projects_5-verify-your-code-locally.html
```bash
[flake8]
max-line-length = 80

```
source-control-for-machine-learning-projects_5-verify-your-code-locally.html
```bash
[flake8]
ignore =
    W504,
    C901,
    E41
max-line-length = 79
exclude =
    .git,
    .cache,
per-file-ignores =
    code/__init__.py:D104
max-complexity = 10
import-order-style = pep8

```
source-control-for-machine-learning-projects_5-verify-your-code-locally.html
```python
# Train the model, return the model
def train_model(data, ridge_args):
    reg_model = Ridge(**ridge_args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

```
source-control-for-machine-learning-projects_5-verify-your-code-locally.html
```python
import numpy as np
from src.model.train import train_model

def test_train_model():
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([10, 9, 8, 8, 6, 5])
    data = {"train": {"X": X_train, "y": y_train}}

    reg_model = train_model(data, {"alpha": 1.2})

    preds = reg_model.predict([[1], [2]])
    np.testing.assert_almost_equal(preds, [9.93939393939394, 9.03030303030303])

```
find-best-classification-model-automated-machine-learning_2-preprocess-data-configure-featurization.html
```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_3-configure-sampling-method.html
```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Choice(values=[0.01, 0.1, 1.0]),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "grid",
    ...
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_3-configure-sampling-method.html
```python
from azure.ai.ml.sweep import Normal, Uniform

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Normal(mu=10, sigma=3),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "random",
    ...
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_3-configure-sampling-method.html
```python
from azure.ai.ml.sweep import RandomParameterSampling

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = RandomParameterSampling(seed=123, rule="sobol"),
    ...
)

```
perform-hyperparameter-tuning-azure-machine-learning-pipelines_3-configure-sampling-method.html
```python
from azure.ai.ml.sweep import Uniform, Choice

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Uniform(min_value=0.05, max_value=0.1),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "bayesian",
    ...
)

```
design-data-ingestion-strategy-for-machine-learning-projects_2-identify-your-data-source-format.html
```bash
{ "deviceId": 29482, "location": "Office1", "time"="2021-07-14T12:47:39Z", "temperature": 23 }

```
deploy-model-managed-online-endpoint_3-deploy-your-mlflow-model-managed-online-endpoint.html
```python
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# create a blue deployment
model = Model(
    path="./model",
    type=AssetTypes.MLFLOW_MODEL,
    description="my sample mlflow model",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()

```
deploy-model-managed-online-endpoint_3-deploy-your-mlflow-model-managed-online-endpoint.html
```python
# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()

```
deploy-model-managed-online-endpoint_3-deploy-your-mlflow-model-managed-online-endpoint.html
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")

```
train-models-training-mlflow-jobs_2-track-metrics-mlflow.html
```yaml
name: mlflow-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - mlflow
    - azureml-mlflow

```
train-models-training-mlflow-jobs_2-track-metrics-mlflow.html
```python
import mlflow

mlflow.autolog()

```
train-models-training-mlflow-jobs_2-track-metrics-mlflow.html
```python
import mlflow

reg_rate = 0.1
mlflow.log_param("Regularization rate", reg_rate)

```
