# Netbalance:

a Systematic Evaluation Framework For Fair Assessment of Bipartite Graph Link Prediction.

## ⚙️ Installation
1. Clone the Repository and Enter the Project Directory:
    ```bash
    git clone https://github.com/sobhanAhmadian/netbalance.git
    cd netbalance
    ```
2. Ensure Correct Python Version
    - Required: >=3.12,<3.13
    - We recommend using a tool like [pyenv](https://github.com/pyenv/pyenv) to manage Python versions.
        ```bash
        pyenv install 3.12
        pyenv local 3.12
        ```
3. We use [Poetry](https://python-poetry.org/) for dependency management. 
    - [Install Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if you haven't already.
    - Configure Poetry to Create Virtual Environment Inside Project Folder (Optional)
        ```bash
        poetry config virtualenvs.in-project true
        ```
4. Create a Virtual Environment and Install Dependencies
    ```bash
    poetry install
    ```
5. Activate the Virtual Environment
    ```bash
    eval $(poetry env activate)
    ```
6. Install Pytorch Manually
    - We do not list PyTorch in pyproject.toml because the package is OS-dependent. [Install PyTorch 2.6.0](https://pytorch.org/get-started/previous-versions/#:~:text=v2.6.0) manually after activating the environment.


## 📖 Tutorials  

We provide a set of Jupyter notebooks demonstrating how to use this project, reproduce results, and explore the methodology step-by-step. You can find them in the [`examples/`](./examples) directory.

### Available Tutorials  
| Notebook | Description |
|----------|-------------|
| [`association_data.ipynb`](./examples/association_data.ipynb) | How to create an association data object, apply different data balancing methodologies, and visualize the results. |
| [`evaluation_framework.ipynb`](./examples/evaluation_framework.ipynb) | How to use netbalnce's evaluation framework to assess the performance of an arbitrary association prediction model on an arbitrary association data. |


## Project Structure
```bash
netbalance/
├── examples/                 
├── src/                      
│   ├── netbalance/           
│   │   ├── __init__.py 
│   │   ├── configs/          # Configuration definitions
│   │   ├── data_repository/  # Repository for data files
│   │   ├── data/             # Data handling
│   │   ├── evaluation/       # Evaluation functions
│   │   ├── features/         # Dataset definitions
│   │   ├── jobs/             # End-point python scripts
│   │   ├── methods/          # Algorithms and methods
│   │   ├── models/           # Model definitions
│   │   ├── optimization/     # Trainer definitions
│   │   ├── utils/            # Utility functions
│   │   └── visualization/    # Visualization functions
├── .env.example    # Example environment variables file
├── .gitignore      # Git ignore file
├── LICENSE         # License file
├── poetry.lock     # Poetry lock file
├── pyproject.toml  # Poetry project file
└── README.md       # Project documentation
```

For example, the files for the **Random Forest model** on the **Sanger** dataset are located at:

+ Dataset Part

    | description | path |
    |-------------|------|
    | Sanger dataset Conigs | [`src/netbalance/configs/sanger.py`](./src/netbalance/configs/sanger.py) |
    | Sanger dataset Handler | [`src/netbalance/features/sanger.py`](./src/netbalance/features/sanger.py) |

+ Model Part
    | description | path |
    |-------------|------|
    | Model and Optimizer Configs | [`src/netbalance/configs/brfsyn.py`](./src/netbalance/configs/brfsyn.py) |
    | Model Defenition | [`src/netbalance/models/brfsyn.py`](./src/netbalance/models/brfsyn.py) |
    | Trainer Definition | [`src/netbalance/optimization/brfsyn.py`](./src/netbalance/optimization/brfsyn.py) |
+ Evaluation Part
    | description | path |
    |-------------|------|
    | Stage 1 | [`src/netbalance/jobs/model_evaluation/calculations/sanger/beta/brfsyn.py`](./src/netbalance/jobs/model_evaluation/calculations/sanger/beta/brfsyn.py) |
    | Stage 2 - Full Test Evaluation | [`src/netbalance/jobs/model_evaluation/results/sanger/beta/eta/brfsyn.py`](./src/netbalance/jobs/model_evaluation/results/sanger/beta/eta/brfsyn.py) |
    | Stage 2 - Balanced Evaluation | [`src/netbalance/jobs/model_evaluation/results/sanger/beta/beta/brfsyn.py`](./src/netbalance/jobs/model_evaluation/results/sanger/beta/beta/brfsyn.py) |
    | Stage 2 - Entity-balanced Evaluation | [`src/netbalance/jobs/model_evaluation/results/sanger/beta/rho/brfsyn.py`](./src/netbalance/jobs/model_evaluation/results/sanger/beta/rho/brfsyn.py) |

## Terminology

The terminology used in this project slightly differs from that in the paper. Below is a brief explanation of the terms used here.

### Evaluation Framework Terminology

| Term | Description |
|------|-------------|
| beta | Equivalent to the term “balanced” used in the paper, it refers to employing a balanced dataset, which can be utilized for either training or testing a model. |
| eta | Equivalent to the term “full test” in the paper, it denotes using the entire dataset. |
| rho | Equivalent to the term “entity-balanced” in the paper, it refers to using a entity-balanced dataset. |
| ibeta | In training, it refers to using balanced data in an iterative manner. |
| irho | In training, it refers to using entity-balanced data in an iterative manner. |

### Model Terminology
| Term | Description |
|------|-------------|
| blindti | Stands for "Base line model - Linear - Drug Target Interaction dataset". It's our linear benchmark model for the drug-target interaction dataset. |
| blinsyn | Stands for "Base line model - Linear - drug SYNergy dataset". It's our linear benchmark model for the drug synergy dataset. |
| bmlpdti | Stands for "Base line model - MLP - Drug Target Interaction dataset". It's our MLP benchmark model for the drug-target interaction dataset. |
| bmlpsyn | Stands for "Base line model - MLP - drug SYNergy dataset". It's our MLP benchmark model for the drug synergy dataset. |
| brfdti | Stands for "Base line model - Random Forest - Drug Target Interaction dataset". It's our random forest benchmark model for the drug-target interaction dataset. |
| brfsyn | Stands for "Base line model - Random Forest - drug SYNergy dataset". It's our random forest benchmark model for the drug synergy dataset. |
| bxgbdti | Stands for "Base line model - XGBoost - Drug Target Interaction dataset". It's our XGBoost benchmark model for the drug-target interaction dataset. |
| bxgbsyn | Stands for "Base line model - XGBoost - drug SYNergy dataset". It's our XGBoost benchmark model for the drug synergy dataset. |