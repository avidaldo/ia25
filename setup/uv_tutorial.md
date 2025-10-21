# uv Tutorial: A Fast Python Package Installer and Resolver

`uv` is an extremely fast Python package installer and resolver, written in Rust. It is designed as a drop-in replacement for `pip` and `venv`.

## 1. Installation

You can install `uv` using `curl` on macOS and Linux, or `pip` if you already have Python installed.

**Using curl (macOS, Linux):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Using PowerShell (Windows):**
```powershell
irm https://astral.sh/uv/install.ps1 | iwr
```

**Using pip:**
```bash
pip install uv
```

Verify the installation:
```bash
uv --version
```

## 2. Managing Virtual Environments

`uv` integrates virtual environment management, similar to `venv`.

### Create a Virtual Environment

To create a new virtual environment, use `uv venv`. By default, it creates a `.venv` directory in your current location.

```bash
# Create a virtual environment in the .venv directory
uv venv
```

You can also specify a Python version if you have multiple versions installed:

```bash
# Create a virtual environment with Python 3.12
uv venv -p 3.12
```

### Activate and Deactivate the Environment

To activate the virtual environment, you need to run the activation script.

**On macOS and Linux:**
```bash
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```batch
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Your terminal prompt will change to indicate that the virtual environment is active.

To deactivate it, simply run:
```bash
deactivate
```

### Removing an Environment

A `uv` virtual environment is just a directory. To remove it, you can delete the directory.

```bash
# Make sure you are not inside the environment
deactivate

# Remove the directory
rm -rf .venv
```

## 3. Initializing a Python Project

Modern `uv` uses `pyproject.toml` to manage dependencies, following PEP 621 standards. This is the recommended approach over `requirements.txt`.

### Initialize a New Project

To create a new Python project with a `pyproject.toml`:

```bash
uv init my-project
cd my-project
```

This creates a basic project structure with a `pyproject.toml` file.

### Initialize in an Existing Directory

If you already have a project directory:

```bash
uv init
```

This will create a `pyproject.toml` in the current directory.

## 4. Managing Packages with `pyproject.toml`

`uv` automatically manages your virtual environment and dependencies when you use `uv add` and `uv remove`.

### Add Packages

Use `uv add` to install packages and automatically update your `pyproject.toml`:

```bash
# Add packages to your project
uv add numpy pandas scikit-learn

# Add a specific version
uv add "numpy==1.26.0"

# Add development dependencies
uv add --dev pytest black
```

This command:
- Creates/activates the virtual environment automatically
- Installs the packages
- Updates `pyproject.toml` with the dependencies
- Creates/updates `uv.lock` for reproducible builds

### Remove Packages

To remove a package:

```bash
uv remove scikit-learn
```

### List Installed Packages

To see all packages in your project:

```bash
uv pip list
```

### Sync Dependencies

To ensure your environment matches `pyproject.toml`:

```bash
uv sync
```

## 5. Running Python with uv

You can run Python scripts and commands without manually activating the virtual environment:

```bash
# Run a Python script
uv run python script.py

# Run a module
uv run python -m pytest

# Run an installed tool
uv run black .
```

`uv run` automatically uses the project's virtual environment.

## 6. Sharing Projects

When sharing your project, include both `pyproject.toml` and `uv.lock` in version control.

### Clone and Setup a Project

Anyone can replicate your environment with:

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install all dependencies (creates venv automatically)
uv sync
```

### Working with Legacy `requirements.txt`

If you need to work with existing `requirements.txt` files:

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Generate requirements.txt from current environment
uv pip freeze > requirements.txt
```

However, migrating to `pyproject.toml` is recommended for better dependency management.
