# Conda Tutorial: Environment and Package Management

Conda is an open-source package and environment management system that runs on Windows, macOS, and Linux. It is widely used in the data science and machine learning communities.

## 1. Installation

We recommend installing **Miniconda**, a minimal installer for conda. It is smaller and faster to install than the full Anaconda distribution.

1.  **Download the installer**: Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.

2.  **Run the installer**: Follow the instructions for your OS. It's recommended to allow the installer to initialize conda, which will make it available in your terminal.

3.  **Verify installation**: Open a new terminal and run:
    ```bash
    conda --version
    ```
    You should see the installed conda version.

## 2. Managing Environments

A conda environment is a directory that contains a specific collection of conda packages that you have installed.

### Create an Environment

To create a new environment, use `conda create`. It's good practice to specify the Python version you want to use.

```bash
# Creates an environment named 'myenv' with Python 3.12
conda create --name myenv python=3.12
```

You can also install packages at the same time:

```bash
conda create --name myenv python=3.12 numpy pandas jupyter
```

### Activate and Deactivate an Environment

Before you can use an environment, you need to activate it.

```bash
# Activate the environment
conda activate myenv
```

Your terminal prompt should change to show the name of the active environment.

To deactivate the current environment and return to the base environment:

```bash
# Deactivate the environment
conda deactivate
```

### List Environments

To see a list of all your environments:

```bash
conda env list
```

The active environment will be marked with an asterisk (*).

### Remove an Environment

To delete an environment and all the packages installed in it:

```bash
# Make sure the environment is not active
conda deactivate

# Remove the environment
conda env remove --name myenv
```

## 3. Managing Packages

With your environment activated, you can install, update, and remove packages.

### Install Packages

Use `conda install` to install packages from the default Anaconda channel.

```bash
# Activate your environment first
conda activate myenv

# Install packages
conda install numpy pandas scikit-learn
```

You can also specify package versions:

```bash
conda install numpy=1.26.0
```

### Install from other channels

Sometimes packages are not available in the default channel. A popular channel is `conda-forge`.

```bash
conda install -c conda-forge some-package
```

### List Installed Packages

To see all packages installed in the current environment:

```bash
conda list
```

### Update Packages

To update a specific package:

```bash
conda update numpy
```

To update all packages in the environment:

```bash
conda update --all
```

### Remove Packages

To uninstall a package:

```bash
conda remove scikit-learn
```

## 4. Sharing Environments

You can share your environment with others by exporting its specification to a YAML file.

### Export an Environment

1.  Activate the environment you want to export.
    ```bash
    conda activate myenv
    ```
2.  Export the environment to a file (commonly named `environment.yml`).
    ```bash
    conda env export > environment.yml
    ```

This file can be shared, and others can replicate your environment.

### Create an Environment from a File

To create an environment from an `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create a new environment with the same name and packages as specified in the file.
