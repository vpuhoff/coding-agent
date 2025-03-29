import subprocess
import os

class SandboxError(Exception):
    pass

class SandboxManager:
    def __init__(self, image: str = "alpine/git"):
        self.image = image

    def run(self, command: str, mount_dir: str) -> str:
        docker_cmd= [
            "docker", "run", "--rm",
            "-v", f"{mount_dir}:/workspace",
            "-w", "/workspace",
            self.image,
            "sh", "-c", command
        ]
        try:
            result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            err_output = e.stderr if e.stderr else e.stdout
            raise SandboxError(f"Command '{command}' failed (exit code {e.returncode}): {err_output.strip()}")
