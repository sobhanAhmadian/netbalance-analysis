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