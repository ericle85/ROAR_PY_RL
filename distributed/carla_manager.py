"""
CARLA server lifecycle manager.

Handles starting, stopping, and restarting CARLA server instances
for distributed workers on Windows.
"""

import os
import subprocess
import time
import socket
from typing import Optional
import logging

from .config import Config, default_config


logger = logging.getLogger(__name__)


class CarlaManager:
    """
    Manages a CARLA server instance.

    Handles:
    - Starting CARLA with correct port
    - Running config.py to configure rendering/sync settings
    - Detecting crashes and restarting
    - Clean shutdown
    """

    def __init__(
        self,
        carla_exe_path: str,
        port: int,
        config: Config = None,
        config_script_path: str = None,
        no_rendering: bool = True,
    ):
        """
        Initialize CARLA manager.

        Args:
            carla_exe_path: Path to CarlaUE4.exe
            port: CARLA server port (e.g., 2000, 2010, 2020)
            config: Configuration object
            config_script_path: Path to config.py script (auto-detected if None)
            no_rendering: Whether to disable rendering
        """
        self.carla_exe_path = carla_exe_path
        self.port = port
        self.config = config or default_config
        self.no_rendering = no_rendering

        # Auto-detect config.py path
        if config_script_path is None:
            # Assume it's in the repo root
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_script_path = os.path.join(repo_root, "config.py")
        self.config_script_path = config_script_path

        self._process: Optional[subprocess.Popen] = None
        self._startup_time: Optional[float] = None

    @property
    def is_running(self) -> bool:
        """Check if CARLA process is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def _is_port_open(self, timeout: float = 1.0) -> bool:
        """Check if CARLA port is accepting connections."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.config.carla_host, self.port))
            sock.close()
            return result == 0
        except socket.error:
            return False

    def start(self, wait_for_ready: bool = True) -> bool:
        """
        Start CARLA server.

        Args:
            wait_for_ready: Whether to wait for CARLA to be ready

        Returns:
            bool: True if started successfully
        """
        if self.is_running:
            logger.warning(f"CARLA already running on port {self.port}")
            return True

        if not os.path.exists(self.carla_exe_path):
            logger.error(f"CARLA executable not found: {self.carla_exe_path}")
            return False

        logger.info(f"Starting CARLA on port {self.port}...")

        try:
            # Start CARLA process
            # Use CREATE_NEW_PROCESS_GROUP on Windows for clean shutdown
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._process = subprocess.Popen(
                [self.carla_exe_path, f"-carla-port={self.port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            self._startup_time = time.time()

            if wait_for_ready:
                return self._wait_for_ready()

            return True

        except Exception as e:
            logger.error(f"Failed to start CARLA: {e}")
            return False

    def _wait_for_ready(self) -> bool:
        """Wait for CARLA to be ready to accept connections."""
        logger.info(f"Waiting for CARLA to be ready (timeout: {self.config.carla_startup_wait}s)...")

        start_time = time.time()
        while time.time() - start_time < self.config.carla_startup_wait:
            if not self.is_running:
                logger.error("CARLA process died during startup")
                return False

            if self._is_port_open():
                logger.info(f"CARLA ready on port {self.port}")

                # Run config.py to set up rendering/sync mode
                if not self._run_config_script():
                    logger.warning("Config script failed, continuing anyway")

                return True

            time.sleep(1.0)

        logger.error(f"CARLA failed to start within {self.config.carla_startup_wait}s")
        return False

    def _run_config_script(self) -> bool:
        """Run config.py to configure CARLA settings."""
        if not os.path.exists(self.config_script_path):
            logger.warning(f"Config script not found: {self.config_script_path}")
            return False

        try:
            cmd = [
                "python",
                self.config_script_path,
                f"--port={self.port}",
            ]

            if self.no_rendering:
                cmd.append("--no-rendering")

            repo_root = os.path.dirname(self.config_script_path)

            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30.0,
            )

            if result.returncode != 0:
                logger.warning(f"Config script returned non-zero: {result.stderr}")
                return False

            logger.info("CARLA configured successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.warning("Config script timed out")
            return False
        except Exception as e:
            logger.warning(f"Config script error: {e}")
            return False

    def stop(self) -> None:
        """Stop CARLA server."""
        if self._process is None:
            return

        logger.info(f"Stopping CARLA on port {self.port}...")

        try:
            # Try graceful termination first
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning("CARLA did not terminate gracefully, killing...")
                self._process.kill()
                self._process.wait(timeout=5.0)
        except Exception as e:
            logger.error(f"Error stopping CARLA: {e}")

        self._process = None
        self._startup_time = None
        logger.info("CARLA stopped")

    def restart(self, wait_for_ready: bool = True) -> bool:
        """
        Restart CARLA server.

        Args:
            wait_for_ready: Whether to wait for CARLA to be ready

        Returns:
            bool: True if restarted successfully
        """
        logger.info("Restarting CARLA...")
        self.stop()
        time.sleep(self.config.worker_restart_delay)
        return self.start(wait_for_ready=wait_for_ready)

    def check_health(self) -> bool:
        """
        Check if CARLA is healthy and responding.

        Returns:
            bool: True if CARLA is healthy
        """
        if not self.is_running:
            return False

        return self._is_port_open(timeout=5.0)

    def ensure_running(self) -> bool:
        """
        Ensure CARLA is running, restarting if necessary.

        Returns:
            bool: True if CARLA is running (possibly after restart)
        """
        if self.check_health():
            return True

        logger.warning("CARLA unhealthy, attempting restart...")
        return self.restart()

    def get_uptime(self) -> Optional[float]:
        """Get uptime in seconds, or None if not running."""
        if self._startup_time is None:
            return None
        return time.time() - self._startup_time

    def __enter__(self):
        """Context manager entry - start CARLA."""
        if not self.start():
            raise RuntimeError(f"Failed to start CARLA on port {self.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop CARLA."""
        self.stop()
        return False


def test_carla_manager():
    """Test CARLA manager (requires CARLA executable)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-exe", required=True, help="Path to CarlaUE4.exe")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    manager = CarlaManager(
        carla_exe_path=args.carla_exe,
        port=args.port,
    )

    print(f"Starting CARLA on port {args.port}...")
    if manager.start():
        print("CARLA started successfully!")
        print(f"Health check: {manager.check_health()}")
        print(f"Uptime: {manager.get_uptime():.1f}s")

        input("Press Enter to stop...")
        manager.stop()
        print("CARLA stopped")
    else:
        print("Failed to start CARLA")


if __name__ == "__main__":
    test_carla_manager()
