import re
import time
import requests
import os
from codecraft.sandbox.sandbox_manager import SandboxManager, SandboxError

class GitError(Exception):
    pass

class GitManager:
    def __init__(self, sandbox_manager: SandboxManager, repo_url: str, base_branch: str, local_path: str, github_token: str = None):
        self.sandbox = sandbox_manager
        self.repo_url = repo_url
        self.base_branch = base_branch
        self.local_path = local_path
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo_owner = None
        self.repo_name = None
        self.current_branch = None
        self._parse_repo_url()

    def _parse_repo_url(self):
        pattern = r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/\.]+)(?:\.git)?"
        match = re.search(pattern, self.repo_url)
        if match:
            self.repo_owner = match.group("owner")
            self.repo_name = match.group("repo")
        else:
            raise GitError("Unsupported repository URL format for GitHub.")

    def clone_repository(self):
        os.makedirs(self.local_path, exist_ok=True)
        clone_url = self.repo_url
        if self.token:
            if clone_url.startswith("http"):
                clone_url = clone_url.replace("https://", f"https://{self.token}@")
            elif clone_url.startswith("git@"):
                part = clone_url.split("github.com:")[-1]
                clone_url = f"https://{self.token}@github.com/{part}"
        if self.base_branch:
            cmd = f"git clone -b {self.base_branch} --single-branch {clone_url} ."
        else:
            cmd = f"git clone {clone_url} ."
        try:
            self.sandbox.run(cmd, self.local_path)
        except SandboxError as e:
            raise GitError(f"Failed to clone repository: {e}")

    def create_branch(self, new_branch_name: str = None) -> str:
        branch = new_branch_name or f"codecraft-ai-{int(time.time())}"
        try:
            self.sandbox.run(f"git checkout -b {branch}", self.local_path)
        except SandboxError as e:
            raise GitError(f"Failed to create branch '{branch}': {e}")
        self.current_branch = branch
        return branch

    def commit_all_changes(self, commit_message: str):
        cmd = (
            'git add -A && '
            'git -c user.name="CodeCraft AI" -c user.email="codecraft@ai" '
            f'commit -m "{commit_message}"'
        )
        try:
            self.sandbox.run(cmd, self.local_path)
        except SandboxError as e:
            if "nothing to commit" in str(e):
                raise GitError("No changes to commit.")
            raise GitError(f"Failed to commit changes: {e}")

    def push_branch(self, branch_name: str = None):
        branch = branch_name or self.current_branch
        if not branch:
            raise GitError("No branch specified to push.")
        try:
            self.sandbox.run(f"git push origin {branch}", self.local_path)
        except SandboxError as e:
            raise GitError(f"Failed to push branch '{branch}': {e}")

    def create_pull_request(self, title: str, body: str = "") -> str:
        if not self.token:
            raise GitError("GitHub token is required to create a pull request.")
        if not self.repo_owner or not self.repo_name:
            raise GitError("Repository owner/name not determined.")
        if not self.current_branch:
            raise GitError("No branch available for pull request.")
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}"
        }
        data = {
            "title": title,
            "head": self.current_branch,
            "base": self.base_branch,
            "body": body
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
        except requests.RequestException as e:
            raise GitError(f"Failed to call GitHub API: {e}")
        if response.status_code not in (200, 201):
            raise GitError(f"Failed to create pull request: {response.status_code} {response.text}")
        pr_url = response.json().get("html_url")
        if not pr_url:
            raise GitError("Pull request created, but no URL returned.")
        return pr_url
