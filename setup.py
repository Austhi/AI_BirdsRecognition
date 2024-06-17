import os
import subprocess
import sys

def create_virtual_env(env_name='venv'):
    print("++++++ SETUP +++++++")

    # Check if the virtual environment directory exists
    if not os.path.exists(env_name):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', env_name])
    else:
        print(f"Virtual environment '{env_name}' already exists.")

def activate_virtual_env(env_name='venv'):
    print("Activating virtual environment...")

    if os.name == 'nt':
        activate_script = os.path.join(env_name, 'Scripts', 'activate.bat')
        command = f"{activate_script} && python -m pip install --upgrade pip"
    else:
        activate_script = os.path.join(env_name, 'bin', 'activate')
        command = f"source {activate_script} && python -m pip install --upgrade pip"

    subprocess.run(command, shell=True, check=True)

def install_requirements(env_name='venv', requirements_file='requirements.txt'):
    print("Installing requirements...")

    if os.name == 'nt':
        pip_executable = os.path.join(env_name, 'Scripts', 'pip')
    else:
        pip_executable = os.path.join(env_name, 'bin', 'pip')

    subprocess.check_call([pip_executable, 'install', '-r', requirements_file])

if __name__ == '__main__':
    create_virtual_env()
    activate_virtual_env()
    install_requirements()