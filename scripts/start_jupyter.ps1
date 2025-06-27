# move to script's directory
Set-Location -Path (Join-Path $PSScriptRoot "..")

# check if conda is installed
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Conda is not installed or not on your PATH."
    Write-Host "Install Miniconda or Anaconda to continue: https://docs.conda.io/"
    Read-Host "Press Enter to exit"
    exit 1
}

# initialize conda for PowerShell
try {
    $condaPath = (Get-Command conda).Source
    $condaRoot = Split-Path (Split-Path $condaPath)
    & "$condaRoot\shell\condabin\conda-hook.ps1"
}
catch {
    Write-Host "Failed to initialize conda. Please run 'conda init powershell' and restart PowerShell."
    Read-Host "Press Enter to exit"
    exit 1
}

$ENV_NAME = "roi-analysis"
$ENV_FILE = "environment.yml"

# check if the environment exists
$envExists = conda info --envs | Select-String "^$ENV_NAME\s"
if (-not $envExists) {
    Write-Host "Environment '$ENV_NAME' not found. Creating it from $ENV_FILE..."
    if (Test-Path $ENV_FILE) {
        conda env create -f $ENV_FILE
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to create environment."
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Host "Error: $ENV_FILE not found in current directory."
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "Environment '$ENV_NAME' already exists."
}

# activate the environment
conda activate $ENV_NAME

# start Jupyter Notebook
jupyter notebook
