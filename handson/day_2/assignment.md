# MLOps on Azure: Day 2 workshop
After setting up the machine learning pipeline during the first workshop, we'll now focus on the CICD and deployment part of the MLOps implementation.

1. Create your own Github repository (does not really matter on which account).
2. Connect it to the Azure ML workspace by creating a service principal. The first two steps of [this](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-github-actions-machine-learning) tutorial will help you with this. <span style="color:red">**Ensure you have a unique name for the service principal**</span>. You can use this resource for subsequent steps as well, but note that the assignment differs from what's discussed there!
3. Productionalize the pipeline creation code (not including the deployment to online endpoint, to avoid some costs and quota limitations)
   1. Convert the solution from day 1 ([notebook](/handson/day_1/handson_solution.ipynb)) to a Python script.
   2. Rename any assets / environments / etc. so it's prefixed with your name
   3. In those same names, also include the environment (`prd`, `uat` or `dev`), which will be passed as a commandline argument (using `argparse`)
   4. If the script is run within the `prd` environment, only schedule the pipeline, but do not run it. If it's any other environment, perform a validation run (i.e. create AND trigger the pipeline).
   5. Put this script in a directory `ml/src/` in your repo. Put the code for the individual components in a subdirectory of that same directory. E.g. `ml/src/components/train/train.py`. The dependency should be put in a subdir called `dependencies`.
4. Create a Github actions pipeline using YAML and put it in `.github/workflows/set_up_pipeline.yml`. In the pipeline:
   1. Set the pipeline to trigger upon changes to `develop` and `main` (it should not trigger on PR creation). Also include the option to trigger it manually (`workflow_dispatch`, note that the option to trigger manually will only appear if the pipeline yaml is part of the default branch).
   2. Create a local variable which, based on the branch the pipeline was triggered for, holds the name of the env (`prd` for `main`, `uat` for `develop` and `dev` for anything else.) (Tip: can use something similar to: [link](https://docs.github.com/en/actions/learn-github-actions/contexts#determining-when-to-use-contexts))
   3. Refer to [link](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-github-actions-machine-learning) again for inspiration on how to clone the repo, and handle the authentication to Azure ML. Note that we'll be using a Python script to set up the ML pipeline.
   4. Set up Python (3.9)
   5. Install the Python SDK v2 (`pip install azure-ai-ml`)
   6. Run your productionalized pipeline script, passing in the local variable which holds the environment.
5. Implement Gitflow safeguards
   1. Create a build validation for `develop` and `main` which will run `pre-commit` on the code when creating a PR. You can use the `.pre-commit-config.yaml` file in the repo (Tip: Use `pre-commit run --all` as a command in the pipeline after installing it through pip `pip install pre-commit`). To add this validation, go to `Settings > Branches`, use `Require status checks to pass before merging` and search for your code check pipeline. Note that you may have to manually run the pipeline once for the job name to show up. Also note that the name you need to search for here is NOT the workflow name, but the `job` name.
   2. While still in the branch permissions, also add the requirement for a PR (to avoid directly pushing to `main` and `develop`). Optionally also add the requirement for a reviewer.
6. Go through the process of making a small change to the code (e.g. use a different model or different hyperparameters) according to Gitflow
   1. Create a feature branch `feature/my_change`
   2. Manually trigger the Github pipeline to validate in `dev`
   3. Create a PR to `develop`, check whether the linting code worked
   4. Approve the PR and validate the `uat` run in Azure ML.
   5. Create a `release` branch (`release/0.1.0`)
   6. Create a PR from release to into `main`
   7. Check that the code check is properly triggered
   8. Approve the PR
   9. Check in Azure ML whether the pipeline schedule was updated