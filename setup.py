import os
import sys
import subprocess
import platform
import re

def is_admin():
    try:
        if platform.system() == 'Windows':
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.geteuid() == 0
    except Exception as e:
        print(f"Unable to check admin rights: {e}")
        return False

def get_python_executable():
    if platform.system() == 'Windows':
        python_executable = os.path.join('venv', 'Scripts', 'python.exe')
    else:
        python_executable = os.path.join('venv', 'bin', 'python')

    return python_executable

def get_pip_executable() -> str:
    if platform.system() == 'Windows':
        pip_executable = os.path.join('venv', 'Scripts', 'pip.exe')
    else:
        pip_executable = os.path.join('venv', 'bin', 'pip')

    return pip_executable

def run_command(command) -> str:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(command)}:")
        print(e.stderr)
        sys.exit(1)

def __get_cuda_version(alternate_path = 'nvidia-smi') -> str:
    try:
        output = run_command([alternate_path])
        match = re.search(r'CUDA Version: (\d+\.\d+)', output)
        if match:
            return match.group(1)
        else:
            return None
    except FileNotFoundError:
        return None

def fix_sudo_nvidia_permissions() -> None:
    try:
        print("Fixing permissions for /dev/nvidia* and /dev/dxg...")
        run_command(['sudo', 'chmod', '666', '/dev/nvidia*', '/dev/dxg'])
        print("Permissions for GPU devices fixed.")
    except SystemExit:
        print("Failed to fix permissions for GPU devices.")

def get_cuda_version() -> str:
    if platform.system() == 'Windows':
        return __get_cuda_version()
    else:
        cuda_version = __get_cuda_version()
        if cuda_version:
            return cuda_version
            
        fix_sudo_nvidia_permissions()
        cudo_version = __get_cuda_version()
        if cuda_version:
            return cuda_version
        
        print("Trying alternate path for nvidia-smi...")
        return __get_cuda_version('/usr/lib/wsl/lib/nvidia-smi')


def get_os():
    return platform.system()

def choose_pytorch_command(os_name, cuda_version):
    pip = get_pip_executable()
    base_command = [pip, "install", "torch", "torchvision", "torchaudio"]

    if os_name == 'Windows':
        os_specific = "windows"
    elif os_name == 'Linux':
        os_specific = "linux"
    elif os_name == 'Darwin':
        os_specific = "macos"
    else:
        print(f"Unknown operating system: {os_name}")
        sys.exit(1)

    if cuda_version:
        cuda_version = ".".join(cuda_version.split('.')[:2])
        cuda_mapping = {
            "12.6": "cu124",
            "12.5": "cu124",
            "12.4": "cu124",
            "12.3": "cu121",
            "12.2": "cu121",
            "12.1": "cu121",
            "11.8": "cu118"
        }

        cuda_tag = cuda_mapping.get(cuda_version)

        if cuda_tag:
            index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
            command = base_command + ["--index-url", index_url]
            print(f"Detected CUDA version: {cuda_version}. Installing PyTorch with CUDA support with tag: {cuda_tag}.")
        else:
            print(f"Unsupported CUDA version: {cuda_version}. Installing CPU-only version.")
            command = base_command
    else:
        print("No NVIDIA GPU detected or nvidia-smi not available. Installing CPU-only version.")
        command = base_command

    return command

def create_venv(venv_path):
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
        print("Virtual environment created.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        sys.exit(1)

def install_dependencies(venv_path, requirements_file):
    print(f"Installing dependencies from {requirements_file}...")
    try:
        pip = get_pip_executable()
        python = get_python_executable()
        
        # Upgrade pip
        subprocess.check_call([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install dependencies
        subprocess.check_call([pip, 'install', '-r', requirements_file])
        print("Dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def install_pytorch(venv_path):
    os_name = get_os()
    cuda_version = get_cuda_version()

    install_command = choose_pytorch_command(os_name, cuda_version)
    print(f"Installing PyTorch using command: {' '.join(install_command)}")

    try:
        subprocess.check_call(install_command)
        print("PyTorch has been successfully installed.")
    except subprocess.CalledProcessError:
        print("An error occurred while installing PyTorch.")
        sys.exit(1)

def main(arg: str):

    if arg == 'create':
        create_venv('venv')
        sys.exit(0)

    if arg != 'install':
        print("Usage: python setup.py [create|install]")
        sys.exit(1)

    venv_path = 'venv'
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"File {requirements_file} not found. Please create it with your dependencies.")
        sys.exit(1)
    
    install_dependencies(venv_path, requirements_file)
    install_pytorch(venv_path)
    
    print("\nVirtual environment has been set up successfully.")
    print("To activate the environment, use the following commands based on your OS and shell:\n")
    
    if platform.system() == 'Windows':
        print("### Command Prompt:")
        print(f"    {venv_path}\\Scripts\\activate.bat\n")
        print("### PowerShell:")
        print(f"    {venv_path}\\Scripts\\Activate.ps1\n")
    else:
        print("### Linux/macOS (bash/zsh):")
        print(f"    source {venv_path}/bin/activate\n")
    
    print("To deactivate the environment, simply run:")
    print("    deactivate\n")

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
