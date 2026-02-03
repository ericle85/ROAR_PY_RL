import subprocess
import time

def start_server(path_to_CarlaEXE: str, port: int):
    subprocess.Popen([path_to_CarlaEXE, f"-carla-port={port}", "-RenderOffScreen"])
    # subprocess.Popen([path_to_CarlaEXE, f"-carla-port={port}"])
    time.sleep(1)
    # subprocess.run(["python", "C:\\Users\\shrek\\ROAR_PY_RL\\config.py", "--no-rendering", "-p", str(port)])
    # subprocess.run(["python", "roar_rl/utils/config.py", "-p", str(port)])
    print(f"Environment started on Port: {port}")


if __name__ == "__main__":
    port = 2002
    start_server(path_to_CarlaEXE=r"C:\Users\shrek\Downloads\Monza_V1.1\Monza\CarlaUE4.exe", port=port)