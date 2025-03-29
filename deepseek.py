import os
import logging
import docker
import git
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Конфигурация
CONFIG = {
    "llm_api_key": os.getenv("LLM_API_KEY"),
    "github_token": os.getenv("GITHUB_TOKEN"),
    "max_debug_attempts": 3,
    "docker_image": "python:3.9-slim",
    "log_level": "INFO"
}

logging.basicConfig(level=CONFIG["log_level"])
logger = logging.getLogger("CodeCraftAI")

class SandboxManager:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def create_sandbox(self) -> str:
        self.container = self.client.containers.run(
            CONFIG["docker_image"],
            command="sleep infinity",
            detach=True,
            remove=True
        )
        return self.container.id

    def exec_command(self, command: str) -> tuple:
        result = self.container.exec_run(command)
        return result.exit_code, result.output.decode()

    def clone_repo(self, repo_url: str, branch: str = "main"):
        cmd = f"git clone -b {branch} {repo_url} /project"
        return self.exec_command(cmd)

    def cleanup(self):
        if self.container:
            self.container.stop()

class LLMService:
    def __init__(self):
        self.base_url = "https://api.llm-provider.com/v1"
        self.headers = {"Authorization": f"Bearer {CONFIG['llm_api_key']}"}

    def analyze_task(self, task: str, code_context: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the development task and codebase:
        Task: {task}
        Code Context: {code_context}
        
        Output JSON format:
        {{
            "affected_components": ["component1", "component2"],
            "required_changes": ["change1", "change2"],
            "potential_issues": ["issue1", "issue2"]
        }}
        """
        response = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt},
            headers=self.headers
        )
        return json.loads(response.json()["content"])

    def generate_code_patch(self, context: str) -> str:
        prompt = f"""
        Generate code changes based on the analysis:
        {context}
        
        Return changes in unified diff format.
        """
        response = requests.post(
            f"{selfbase_url}/generate",
            json={"prompt": prompt},
            headers=self.headers
        )
        return response.json()["content"]

class CodeAnalyzer:
    def __init__(self, sandbox: SandboxManager):
        self.sandbox = sandbox

    def detect_tech_stack(self) -> Dict[str, str]:
        tech_stack = {}
        # Анализ файлов конфигурации
        exit_code, output = self.sandbox.exec_command("ls /project")
        if "package.json" in output:
            tech_stack["frontend"] = "React"
        if "requirements.txt" in output:
            tech_stack["backend"] = "Django"
        return tech_stack

    def find_entry_points(self) -> list:
        # Используем LLM для анализа структуры
        _, code = self.sandbox.exec_command("find /project -type f")
        analysis = LLMService().analyze_task("Find entry points", code)
        return analysis.get("entry_points", [])

class VerificationRunner:
    def __init__(self, sandbox: SandboxManager):
        self.sandbox = sandbox

    def run_tests(self) -> bool:
        exit_code, output = self.sandbox.exec_command("npm test" if "package.json" 
                                                    else "pytest")
        return exit_code == 0

    def run_linter(self) -> bool:
        exit_code, _ = self.sandbox.exec_command("eslint ." if "package.json" 
                                               else "flake8 .")
        return exit_code == 0

class VCSManager:
    def __init__(self, sandbox: SandboxManager):
        self.sandbox = sandbox

    def create_branch(self, branch_name: str):
        self.sandbox.exec_command(f"git checkout -b {branch_name}")

    def create_pr(self, repo: str, title: str, description: str) -> str:
        pr_data = {
            "title": title,
            "head": branch_name,
            "base": "main",
            "body": description
        }
        response = requests.post(
            f"https://api.github.com/repos/{repo}/pulls",
            headers={"Authorization": f"token {CONFIG['github_token']}"},
            json=pr_data
        )
        return response.json()["html_url"]

class Orchestrator:
    def __init__(self):
        self.sandbox = SandboxManager()
        self.llm = LLMService()
        self.current_state = {}

    def execute_task(self, task: str, repo_url: str):
        try:
            # Этап 1: Подготовка окружения
            self.sandbox.create_sandbox()
            self.sandbox.clone_repo(repo_url)
            
            # Этап 2: Анализ
            analyzer = CodeAnalyzer(self.sandbox)
            tech_stack = analyzer.detect_tech_stack()
            entry_points = analyzer.find_entry_points()
            
            # Этап 3: Генерация изменений
            code_context = f"Tech Stack: {tech_stack}\nEntry Points: {entry_points}"
            analysis = self.llm.analyze_task(task, code_context)
            patch = self.llm.generate_code_patch(analysis)
            
            # Применение изменений
            self.apply_patch(patch)
            
            # Этап 4: Верификация
            verifier = VerificationRunner(self.sandbox)
            if not verifier.run_linter():
                raise Exception("Linter errors")
            
            if not verifier.run_tests():
                raise Exception("Test failures")
            
            # Создание PR
            vcs = VCSManager(self.sandbox)
            pr_url = vcs.create_pr(repo_url.split("/")[-1], 
                                 f"Implement: {task}", 
                                 analysis["change_description"])
            
            logger.info(f"PR created: {pr_url}")
            return pr_url
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
        finally:
            self.sandbox.cleanup()

    def apply_patch(self, patch: str):
        # Сохраняем патч и применяем с помощью git
        self.sandbox.exec_command("echo '{}' > /project/changes.diff".format(patch))
        exit_code, _ = self.sandbox.exec_command(
            "git apply changes.diff", 
            workdir="/project"
        )
        if exit_code != 0:
            raise Exception("Failed to apply patch")

# Пример использования
if __name__ == "__main__":
    agent = Orchestrator()
    pr_link = agent.execute_task(
        task="Add user registration form with validation",
        repo_url="https://github.com/example/repo"
    )
    print(f"Created PR: {pr_link}")