import subprocess
import time

def start_server(path_to_CarlaEXE: str, port: int):
   # Process stops after some time
   subprocess.Popen([path_to_CarlaEXE, f"-carla-port={port}"])
   time.sleep(1)


   # Run (roar_competition) shrek@MINIPC C:\Users\shrek\ROAR_PY_RL>python config.py --no-rendering --port 2002
   subprocess.run([
       "python",
       "config.py",
       "--no-rendering",
       f"--port={port}"
       
   ], cwd=r"C:\Users\shrek\ROAR_PY_RL")
   print(f"Environment started on Port: {port}")


if __name__ == "__main__":
   port = 2002
   start_server(path_to_CarlaEXE=r"C:\Users\shrek\Downloads\Monza_V1.1\Monza\CarlaUE4.exe", port=port)