# challenge-welding-reference-solution-1

This repository contains the code of a naive reference solution proposed for the Welding Quality Detection challenge. This solution is provided as a pedagogical example only and should not be considered a benchmark in terms of performance.

# Preparing your environnement

## Create and activate your virtual environnement
To create a virtual environnement, you can use many different tools as python virtual environments (venv), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [uv](https://github.com/astral-sh/uv). Conda and uv has the advantage of making possible to choose the python version  you want to use in your virtual env by setting X.Y in the commands below:

If using python integrated venv , to create a new virtual environnement, type in a terminal : 

 ```commandline
 python -m venv path_to_your_env
 ``` 

If using conda : 

```commandline
conda create -n path_to_your_env python=X.Y 
```

If using uv :

```commandline
uv venv path_to_your_env --python=X.Y
```

## Activate your virtual env

On Windows power shell 
```commandline
./path_to_your_env/Scripts/activate
```

On Linux: 
```commandline
source path_to_your_env/bin/activate
```

## Installation of the ChallengeWelding package
To install the package Challenge Welding and its dependencies, from the root directory of this repository type:  
```commandline 
pip install .
```

```commandline 
pip install -r requirements.txt
```

# Reference Solution content
The scripts and Jupyter notebooks are provided to guide participants through the different utilities developed to facilitate dataset manipulation and AI component development and evaluation:

- ```01-Tutorial.py``` : This script demonstrates how to use the main user functions present in this package. It includes examples of how to list available datasets, explore metadata, and draw basic statistics on contextual variables.

- ```02-Create_pytorch_dataloader.py``` : This script shows how to use the package to create a PyTorch dataloader.

- ```03-Test_AIComponent.py``` : This script demonstrates how to load an AI component and evaluate it by generating operational and uncertainty metrics. 

These scripts are also available as jupyter notebooks, where we provide more information concerning the use case, the dataset, and the AI component to be developed and evaluated: 

- ```01-Tutorial.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/01-Tutorial.ipynb) 

- ```02-Create_pytorch_dataloader.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/02-Create_pytorch_dataloader.ipynb) 

- ```03-Test_AIComponent.ipynb``` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/confianceai/Challenge-Welding-Starter-Kit/blob/main/examples/03-Test_AIComponent.ipynb) 
