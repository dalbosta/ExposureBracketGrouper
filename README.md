# ExposureBracketGrouper
Groups unlabeled images shot at the same camera angle with varying exposures



# Project Setup Instructions

## Prerequisites
- Python 3.12.8 installed on your system
- pip package manager

## Setup Steps

### 1. Create Virtual Environment
Make sure you're in the project root directory, then create a virtual environment with Python 3.12.8:

```bash
# Windows
python3.12 -m venv venv
# OR using py launcher
py -3.12 -m venv venv

# Unix/MacOS
python3.12 -m venv venv
```

### 2. Activate Virtual Environment

```bash
# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Unix/MacOS
source venv/bin/activate
```

### 3. Upgrade pip
Once the virtual environment is activated, upgrade pip to the latest version:

```bash
python -m pip install --upgrade pip
```

### 4. Install Requirements
Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### Verify Installation
To verify that everything is installed correctly:
```bash
python --version  # Should show Python 3.12.8
pip list         # Should show all installed packages
```

## Troubleshooting
- If you don't see `(venv)` at the start of your command prompt, the virtual environment is not activated
- If you encounter permission issues on Windows when activating the virtual environment, you might need to adjust PowerShell execution policies
- Make sure you're using the correct Python version when creating the virtual environment
