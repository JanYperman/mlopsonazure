* Install VSCode (or another IDE, but the integration with VSCode is pretty good)
* Have access to a Bash command line (PowerShell will probably work as well, but not sure if I'll be able to help you)
* Install the Azure CLI: [instructions](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
* Log into the Azure CLI in your command line: `az login`
* Install Azure Machine learning CLI extension: [instructions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)
* Install Anaconda: [instructions](https://docs.anaconda.com/free/anaconda/install/index.html) (Azure ML mainly uses anaconda to specify environments)
* Create a conda virtual env with the Python SDK version 2 installed. You can paste the following into a file named `env.yml`:
    ```yaml
    name: handson_aml

    dependencies:
    - jupyter
    - pip:
        - azure-ai-ml
    ```
    and run `conda env create -f env.yml`
    This will create a conda env named `handson_aml`.
* Check whether you have access to the Azure workspace for the training. You can check by e.g. running the following Azure ML CLI command:
    ```bash
    az ml data list --subscription 59a62e46-b799-4da2-8314-f56ef5acf82b -g rg-azuremltraining -w dummy-workspace
    ```
    which should list the data assets (don't worry about what this is for now).
