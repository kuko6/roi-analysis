@echo off
setlocal

REM move to script's directory
cd /d "%~dp0\.."

REM check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda is not installed or not on your PATH.
    echo Install Miniconda or Anaconda to continue: https://docs.conda.io/
    pause
    exit /b 1
)

set ENV_NAME=roi-analysis
set ENV_FILE=environment.yml

REM check if the environment exists
conda info --envs | findstr /b "%ENV_NAME%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Environment '%ENV_NAME%' not found. Creating it from %ENV_FILE%...
    if exist "%ENV_FILE%" (
        conda env create -f "%ENV_FILE%"
    ) else (
        echo Error: %ENV_FILE% not found in current directory.
        pause
        exit /b 1
    )
) else (
    echo Environment '%ENV_NAME%' already exists.
)

REM activate the environment
call conda activate "%ENV_NAME%"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate environment.
    pause
    exit /b 1
)

REM start Jupyter Notebook
jupyter notebook
call conda activate "%ENV_NAME%"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to start jupyter.
    pause
    exit /b 1
)
