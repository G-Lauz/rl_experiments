# Experiments with reinforcement learning

## Quick start
 ```Bash
 git submodule update --init --recursive
 
 python -m venv .venv

 .\.venv\Scripts\Activate.ps1 # Powershell
 .\.venv\Scripts\activate.bat # Windows cmd
 source .venv/bin/activate # Ubuntu

 python -m pip install --upgrade pip
 
 pip install -e .
 pip install -r ./requirements/dev-requirements.txt
 ```

 **Note:** You might need the following for `Powershell`:
 ```Bash
 Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
 ```

 **Note 2:** To use GPUs with PyTorch, you should download the required package according to your needs from https://pytorch.org/ and make sure to replace the version installed with the [requirements.txt](requirements/requirements.txt).
