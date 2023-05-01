# General reference
The main documentation can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2).
Exam will be using SDK v2: [link](https://trainingsupport.microsoft.com/en-us/mcp/forum/all/dp-100-python-sdk-v1-or-v2/3a5d91ad-a94f-4f19-bcda-4aed74017242)
# Design and prepare a machine learning solution (20–25%)
## Design a machine learning solution
> Design a machine learning solution: [link](https://learn.microsoft.com/en-us/training/paths/design-machine-learning-solution/)
* Determine the appropriate compute specifications for a training workload
* Describe model deployment requirements
* Select which development approach to use to build or train a model
## Manage an Azure Machine Learning workspace
> Explore the Azure Machine Learning workspace: [link](https://learn.microsoft.com//en-us/training/paths/explore-azure-machine-learning-workspace/)
* Create an Azure Machine Learning workspace
* Manage a workspace by using developer tools for workspace interaction
* <span style="color:red">Set up Git integration for source control</span>.
  > Maybe in the MLOps learning path?
## Manage data in an Azure Machine Learning workspace
> Work with data in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/work-data-azure-machine-learning/)
* Select Azure Storage resources
* Register and maintain datastores
* Create and manage data assets
## Manage compute for experiments in Azure Machine Learning
> Partial: [link](https://learn.microsoft.com/en-us/training/modules/explore-azure-machine-learning-workspace-resources-assets/3-identify-resources). Is largely scattered across the difference learning paths.
Can look into "Work with compute" [link](https://microsoftlearning.github.io/mslearn-azure-ml/Instructions/04-Work-with-compute.html).
* Create compute targets for experiments and training
* Select an environment for a machine learning use case
* Configure attached compute resources, including Apache Spark pools
* Monitor compute utilization
# Explore data and train models (35–40%)
## Explore data by using data assets and data stores
> Work with data in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/work-data-azure-machine-learning/)
* Access and wrangle data during interactive development
* Wrangle interactive data with Apache Spark.
  > Not covered in learning paths, use [this](https://learn.microsoft.com/en-us/azure/machine-learning/interactive-data-wrangling-with-apache-spark-azure-ml?view=azureml-api-2) instead.
## Create models by using the Azure Machine Learning designer
> Explore visual tools for machine learning: [link](https://learn.microsoft.com/en-us/training/paths/create-no-code-predictive-models-azure-machine-learning/)
* Create a training pipeline
* Consume data assets from the designer
* Use custom code components in designer
* Evaluate the model, including responsible AI guidelines.
  > [link](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai)
## Use automated machine learning to explore optimal models
> Automate machine learning model selection with Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/automate-machine-learning-model-selection-azure-machine-learning/)
Note: This only covers classification!
* Use automated machine learning for tabular data
* Use automated machine learning for computer vision
  > [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2)
* Use automated machine learning for natural language processing (NLP)
  > [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-nlp-models?view=azureml-api-2)
* Select and understand training options, including preprocessing and algorithms
* Evaluate an automated machine learning run, including responsible AI guidelines.
  > [link](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai)
## Use notebooks for custom model training
> Train models with scripts in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/train-models-scripts-azure-machine-learning/)
* Develop code by using a compute instance
* Track model training by using MLflow
* Evaluate a model
* Train a model by using Python SDKv2
* Use the terminal to configure a compute instance
## Tune hyperparameters with Azure Machine Learning
> Optimize model training in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/modules/perform-hyperparameter-tuning-azure-machine-learning-pipelines/)
* Select a sampling method
* Define the search space
* Define the primary metric
* Define early termination options
# Prepare a model for deployment (20–25%)
## Run model training scripts
> Train models with scripts in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/train-models-scripts-azure-machine-learning/)
* Configure job run settings for a script
* Configure compute for a job run
* Consume data from a data asset in a job
* Run a script as a job by using Azure Machine Learning
* Use MLflow to log metrics from a job run
* Use logs to troubleshoot job run errors
* Configure an environment for a job run
* Define parameters for a job
## Implement training pipelines
> Optimize model training in Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/use-azure-machine-learning-pipelines-for-automation/)
* Create a pipeline
* Pass data between steps in a pipeline
* Run and schedule a pipeline
* Monitor pipeline runs
* Create custom components
* Use component-based pipelines
## Manage models in Azure Machine Learning
> MLFlow model not covered very well in learning paths, mostly the focus is on tracking metrics etc.
Can refer to this instead: [link](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow-models?view=azureml-api-2)
* Describe MLflow model output
* Identify an appropriate framework to package a model
* Assess a model by using responsible AI guidelines
  > [link](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai)
# Deploy and retrain a model (10–15%)
## Deploy a model
> Deploy and consume models with Azure Machine Learning: [link](https://learn.microsoft.com//en-us/training/paths/deploy-consume-models-azure-machine-learning/)
* Configure settings for online deployment
* Configure compute for a batch deployment
* Deploy a model to an online endpoint
* Deploy a model to a batch endpoint
* Test an online deployed service
* Invoke the batch endpoint to start a batch scoring job
## Apply machine learning operations (MLOps) practices
> MLOps: [link](https://learn.microsoft.com/en-us/training/paths/introduction-machine-learn-operations/)
* Trigger an Azure Machine Learning job, including from Azure DevOps or GitHub
* Automate model retraining based on new data additions or data changes
  > This isn't supported in SDK v2? Could use Data Factory instead, but feels outside of scope [link](https://learn.microsoft.com/en-us/azure/data-factory/how-to-create-event-trigger?tabs=data-factory)
* Define event-based retraining triggers
  > Not sure what they're getting at here. Could be referring to event grid: [link](https://learn.microsoft.com/en-us/azure/event-grid/overview?view=azureml-api-2), [link](https://learn.microsoft.com/en-us/training/modules/azure-event-grid/).
  [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-event-grid-batch?view=azureml-api-2&tabs=cli) is an example of batch inference setting up Event Grid + Logic app
