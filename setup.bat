@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call .\venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found. Skipping dependency installation.
)

echo Setup complete!
pause