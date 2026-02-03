import subprocess

# The name of the process you want to kill (CarlaUE4.exe in your case)
process_name = "CarlaUE4.exe"

# Construct the taskkill command
command = f'taskkill /IM {process_name} /F'

try:
    # Run the command using subprocess
    subprocess.run(command, shell=True, check=True)
    print(f"Successfully killed the process {process_name}")
except subprocess.CalledProcessError as e:
    print(f"Failed to kill the process: {e}")
