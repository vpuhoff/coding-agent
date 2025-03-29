# requirements.txt:
# google-generativeai
# python-dotenv
# docker
# GitPython
# requests

import os
import sys
import subprocess
import shutil
import logging
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import google.generativeai as genai
from dotenv import load_dotenv
import docker
from docker.errors import DockerException, NotFound, APIError
from docker.models.containers import Container
import git
from git.exc import GitCommandError
import requests

# --- Configuration ---

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("CodeCraftAI")

@dataclass
class Config:
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN") # Or GitLab token
    vcs_provider: str = "github" # or "gitlab"
    docker_image: str = "python:3.10-slim" # Base image, may need project-specific ones
    sandbox_base_dir: str = "./sandbox"
    max_debug_iterations: int = 5
    llm_model_name: str = "gemini-1.5-flash" # Or another suitable model
    # Default commands (can be overridden by project detection or config)
    default_install_commands: List[str] = field(default_factory=lambda: ["pip install -r requirements.txt", "npm install"])
    default_build_commands: List[str] = field(default_factory=lambda: ["npm run build"]) # Example
    default_lint_commands: List[str] = field(default_factory=lambda: ["npm run lint", "flake8 .", "pylint ."]) # Example
    default_test_commands: List[str] = field(default_factory=lambda: ["npm test", "pytest"]) # Example

    def __post_init__(self):
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        if not self.github_token and self.vcs_provider in ["github", "gitlab"]:
             logger.warning(f"{self.vcs_provider.upper()}_TOKEN not found in environment variables. PR creation will fail.")
        genai.configure(api_key=self.gemini_api_key)

CONFIG = Config()

# --- State Management ---

@dataclass
class TaskState:
    task_description: str
    repo_url: str
    base_branch: str
    target_branch: str = ""
    sandbox_id: Optional[str] = None
    container: Optional[Container] = None
    repo_path_in_sandbox: str = "/workspace"
    project_type: Optional[str] = None
    install_commands: List[str] = field(default_factory=list)
    build_commands: List[str] = field(default_factory=list)
    lint_commands: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    relevant_files: List[str] = field(default_factory=list)
    coding_patterns: Optional[str] = None
    current_patch: Optional[str] = None
    debug_iteration: int = 0
    last_error_output: Optional[str] = None
    pr_url: Optional[str] = None
    status: str = "pending" # pending, analyzing, generating, verifying, debugging, testing, committing, completed, failed
    log: List[str] = field(default_factory=list)

    def add_log(self, message: str):
        logger.info(message)
        self.log.append(message)

# --- Exceptions ---

class CodeCraftError(Exception):
    """Base exception for CodeCraftAI errors."""
    pass

class SandboxError(CodeCraftError):
    """Error related to sandbox operations."""
    pass

class VCSError(CodeCraftError):
    """Error related to version control operations."""
    pass

class LLMError(CodeCraftError):
    """Error related to LLM interactions."""
    pass

class VerificationError(CodeCraftError):
    """Error during build, lint, or test verification."""
    pass

# --- Component Implementations ---

class SandboxManager:
    """Manages Docker containers for isolated execution."""

    def __init__(self, config: Config):
        self.config = config
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("Docker client initialized successfully.")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise SandboxError(f"Docker is not running or accessible: {e}")
        Path(self.config.sandbox_base_dir).mkdir(parents=True, exist_ok=True)

    def _generate_container_name(self) -> str:
        return f"codecraft-sandbox-{uuid.uuid4().hex[:8]}"

    def create_sandbox(self, state: TaskState) -> Tuple[str, Container]:
        """Creates and starts a new Docker container."""
        container_name = self._generate_container_name()
        state.add_log(f"Creating sandbox container '{container_name}' using image '{self.config.docker_image}'...")
        try:
            # Pull the image if it doesn't exist locally
            try:
                self.client.images.get(self.config.docker_image)
                logger.info(f"Docker image '{self.config.docker_image}' found locally.")
            except docker.errors.ImageNotFound:
                logger.info(f"Pulling Docker image '{self.config.docker_image}'...")
                self.client.images.pull(self.config.docker_image)
                logger.info(f"Successfully pulled Docker image '{self.config.docker_image}'.")

            container = self.client.containers.run(
                image=self.config.docker_image,
                command="/bin/sh -c 'while true; do sleep 3600; done'", # Keep container running
                name=container_name,
                working_dir=state.repo_path_in_sandbox,
                volumes={
                    # We'll copy code in later rather than mounting directly
                    # This avoids potential host FS pollution if agent goes wrong
                },
                detach=True,
                auto_remove=False, # Keep it for inspection if needed, cleanup later
                user=f"{os.getuid()}:{os.getgid()}" # Run as current user if possible
            )
            state.add_log(f"Sandbox container '{container.id}' created and started.")
            state.sandbox_id = container.id
            state.container = container
             # Ensure the workspace directory exists
            self.run_command(state, f"mkdir -p {state.repo_path_in_sandbox}")
            # Install git inside the container
            self.run_command(state, "apt-get update && apt-get install -y git || apk update && apk add git")

            return container.id, container
        except (APIError, DockerException) as e:
            logger.error(f"Failed to create sandbox container: {e}")
            raise SandboxError(f"Failed to create sandbox: {e}")

    def run_command(self, state: TaskState, command: str, workdir: Optional[str] = None, stream_output: bool = False) -> Tuple[int, str]:
        """Runs a command inside the specified sandbox container."""
        if not state.container:
            raise SandboxError("Sandbox container not available.")

        effective_workdir = workdir or state.repo_path_in_sandbox
        full_command = f"sh -c 'cd {effective_workdir} && {command}'"
        state.add_log(f"Running command in sandbox '{state.sandbox_id}' (dir: {effective_workdir}): {command}")

        try:
            exit_code, output_stream = state.container.exec_run(full_command, stream=True, demux=False, user='root') # Run install/setup as root

            output = ""
            for chunk in output_stream:
                decoded_chunk = chunk.decode('utf-8', errors='replace')
                if stream_output:
                    print(decoded_chunk, end='') # Print intermediate output
                output += decoded_chunk

            # Re-run exec_inspect to get the final exit code accurately after stream processing
            exec_instance_id = state.container.exec_run(full_command, stream=False, demux=False)['Id']
            exec_info = self.client.api.exec_inspect(exec_instance_id)
            final_exit_code = exec_info['ExitCode']


            if final_exit_code == 0:
                state.add_log(f"Command finished successfully (Exit Code: {final_exit_code}). Output head: {output[:200]}...")
            else:
                 state.add_log(f"Command failed (Exit Code: {final_exit_code}). Output: {output}")
            return final_exit_code, output

        except APIError as e:
            logger.error(f"API error running command '{command}' in sandbox '{state.sandbox_id}': {e}")
            raise SandboxError(f"API error executing command: {e}")
        except Exception as e:
            logger.error(f"Unexpected error running command '{command}' in sandbox '{state.sandbox_id}': {e}")
            raise SandboxError(f"Unexpected error executing command: {e}")


    def copy_to_sandbox(self, state: TaskState, src_path: str, dest_path: str):
        """Copies files/directories from host to sandbox."""
        if not state.container:
            raise SandboxError("Sandbox container not available.")
        state.add_log(f"Copying host:'{src_path}' to sandbox:'{state.sandbox_id}:{dest_path}'")
        # Docker SDK copy is tricky, using 'docker cp' CLI via subprocess is often more reliable
        try:
            subprocess.run(
                ["docker", "cp", src_path, f"{state.sandbox_id}:{dest_path}"],
                check=True, capture_output=True, text=True
            )
            state.add_log(f"Successfully copied '{src_path}' to sandbox.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy to sandbox: {e}\nStderr: {e.stderr}\nStdout: {e.stdout}")
            raise SandboxError(f"Failed 'docker cp' from host to sandbox: {e.stderr}")
        except FileNotFoundError:
             logger.error("Failed to copy to sandbox: 'docker' command not found. Is Docker installed and in PATH?")
             raise SandboxError("'docker' command not found.")

    def copy_from_sandbox(self, state: TaskState, src_path: str, dest_path: str):
        """Copies files/directories from sandbox to host."""
        if not state.container:
            raise SandboxError("Sandbox container not available.")
        state.add_log(f"Copying sandbox:'{state.sandbox_id}:{src_path}' to host:'{dest_path}'")
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["docker", "cp", f"{state.sandbox_id}:{src_path}", dest_path],
                check=True, capture_output=True, text=True
            )
            state.add_log(f"Successfully copied from sandbox to '{dest_path}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy from sandbox: {e}\nStderr: {e.stderr}\nStdout: {e.stdout}")
            raise SandboxError(f"Failed 'docker cp' from sandbox to host: {e.stderr}")
        except FileNotFoundError:
             logger.error("Failed to copy from sandbox: 'docker' command not found.")
             raise SandboxError("'docker' command not found.")

    def cleanup_sandbox(self, state: TaskState):
        """Stops and removes the sandbox container."""
        if state.container:
            container_id = state.container.id
            state.add_log(f"Cleaning up sandbox container '{container_id}'...")
            try:
                state.container.stop(timeout=10)
                state.container.remove(force=True)
                state.add_log(f"Sandbox container '{container_id}' stopped and removed.")
            except (NotFound, APIError, DockerException) as e:
                logger.warning(f"Could not cleanup sandbox container '{container_id}': {e}")
            finally:
                state.container = None
                state.sandbox_id = None
        else:
            state.add_log("No sandbox container to clean up.")


class VCSManager:
    """Handles Git operations and interacts with VCS hosting providers."""

    def __init__(self, config: Config, sandbox_manager: SandboxManager):
        self.config = config
        self.sandbox_manager = sandbox_manager

    def _run_git_command(self, state: TaskState, git_command: str) -> str:
        """Executes a git command inside the sandbox."""
        full_command = f"git {git_command}"
        exit_code, output = self.sandbox_manager.run_command(state, full_command, workdir=state.repo_path_in_sandbox)
        if exit_code != 0:
            raise VCSError(f"Git command failed: 'git {git_command}'. Output: {output}")
        return output

    def clone_repo(self, state: TaskState):
        """Clones the repository into the sandbox."""
        state.add_log(f"Cloning repository '{state.repo_url}' into sandbox...")
        try:
            # Use token for private repos if available
            repo_url_with_auth = state.repo_url
            if self.config.github_token and "github.com" in repo_url_with_auth:
                 repo_url_with_auth = repo_url_with_auth.replace("https://", f"https://oauth2:{self.config.github_token}@")
            # Add other providers (GitLab etc.) if needed

            self._run_git_command(state, f"clone --branch {state.base_branch} --depth 1 {repo_url_with_auth} {state.repo_path_in_sandbox}")
            state.add_log(f"Repository cloned successfully into {state.repo_path_in_sandbox}")
        except VCSError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise VCSError(f"Failed to clone {state.repo_url}: {e}")

    def create_branch(self, state: TaskState):
        """Creates a new branch for the task."""
        state.target_branch = f"codecraft-ai/{uuid.uuid4().hex[:8]}"
        state.add_log(f"Creating new branch '{state.target_branch}' from '{state.base_branch}'...")
        try:
            self._run_git_command(state, f"checkout -b {state.target_branch}")
            state.add_log(f"Switched to new branch '{state.target_branch}'.")
        except VCSError as e:
            logger.error(f"Failed to create or switch branch: {e}")
            raise VCSError(f"Failed to create branch {state.target_branch}: {e}")

    def apply_patch(self, state: TaskState, patch_content: str) -> bool:
        """Applies a patch (diff format) inside the sandbox."""
        if not patch_content:
            state.add_log("Patch content is empty, skipping application.")
            return True

        state.add_log("Applying generated patch...")
        patch_filename = f"/tmp/codecraft_{uuid.uuid4().hex[:8]}.patch"
        # Write patch content to a file inside the container
        # Using echo with careful escaping is simpler than copying a temp file
        # Ensure the patch content doesn't contain sequences that break the shell command
        escaped_patch = patch_content.replace("'", "'\\''") # Basic escaping for single quotes
        write_command = f"echo '{escaped_patch}' > {patch_filename}"
        exit_code, _ = self.sandbox_manager.run_command(state, write_command, workdir="/")
        if exit_code != 0:
            state.add_log("Failed to write patch file inside sandbox.")
            return False

        # Try applying the patch
        # `git apply` is generally safer for applying external patches
        # --reject allows applying parts that succeed and shows rejects for failures
        # --whitespace=fix can help with minor whitespace issues
        apply_command = f"git apply --reject --whitespace=fix {patch_filename}"
        exit_code, output = self.sandbox_manager.run_command(state, apply_command, workdir=state.repo_path_in_sandbox)

        # Clean up patch file regardless of outcome
        self.sandbox_manager.run_command(state, f"rm {patch_filename}", workdir="/")

        if exit_code == 0:
            state.add_log(f"Patch applied successfully. Output:\n{output}")
            # Check for .rej files (rejected hunks)
            exit_code_rej, output_rej = self.sandbox_manager.run_command(state, "find . -name '*.rej' -type f", workdir=state.repo_path_in_sandbox)
            if output_rej.strip():
                 state.add_log(f"Patch applied with rejects (.rej files created):\n{output_rej}")
                 # Consider failing here or letting subsequent steps catch issues
                 # For now, log warning and proceed
                 logger.warning(f"Patch applied but rejects were generated: {output_rej}")
                 # Maybe try to add the .rej content to the error message for LLM?
                 # state.last_error_output = output + "\nRejects found:\n" + output_rej # Example
            return True
        else:
            state.add_log(f"Failed to apply patch. Exit Code: {exit_code}. Output:\n{output}")
            state.last_error_output = f"Failed to apply patch:\n{output}"
            return False


    def commit_changes(self, state: TaskState, commit_message: str):
        """Stages all changes and commits them."""
        state.add_log("Staging changes...")
        try:
            # Stage all changes (new, modified, deleted)
            self._run_git_command(state, "add -A")
            state.add_log("Changes staged.")

            state.add_log(f"Committing changes with message: '{commit_message}'")
            # Need to configure git user inside container if not done globally
            self._run_git_command(state, "config user.email 'codecraft-ai@example.com'")
            self._run_git_command(state, "config user.name 'CodeCraft AI Agent'")
            # Use -m to pass message directly, avoiding editor issues
            self._run_git_command(state, f"commit -m \"{commit_message.replace('\"', '\\\"')}\"") # Escape quotes
            state.add_log("Changes committed successfully.")
        except VCSError as e:
            logger.error(f"Failed to commit changes: {e}")
            # Check if maybe there were no changes to commit
            if "nothing to commit" in str(e).lower():
                 state.add_log("No changes detected to commit.")
            else:
                raise VCSError(f"Failed to commit: {e}")


    def push_branch(self, state: TaskState):
        """Pushes the current branch to the remote origin."""
        state.add_log(f"Pushing branch '{state.target_branch}' to origin...")
        try:
            # Ensure remote origin URL uses token if needed (might be already set by clone)
             # Set upstream branch on first push
            self._run_git_command(state, f"push --set-upstream origin {state.target_branch}")
            state.add_log(f"Branch '{state.target_branch}' pushed successfully.")
        except VCSError as e:
            logger.error(f"Failed to push branch: {e}")
            raise VCSError(f"Failed to push {state.target_branch}: {e}")

    def create_pull_request(self, state: TaskState, title: str, body: str) -> Optional[str]:
        """Creates a pull request on GitHub/GitLab."""
        if not self.config.github_token: # Adjust for GitLab token variable if needed
            state.add_log("VCS token not configured. Skipping Pull Request creation.")
            return None

        state.add_log(f"Creating Pull Request for branch '{state.target_branch}' against '{state.base_branch}'...")

        # Extract owner and repo name from URL
        match = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/.]+?)(?:\.git)?$", state.repo_url)
        if not match:
             # Add GitLab regex or other providers if needed
             logger.error(f"Could not parse owner/repo from URL: {state.repo_url}")
             raise VCSError(f"Cannot parse owner/repo from URL {state.repo_url} for PR creation.")
        owner, repo = match.groups()

        if self.config.vcs_provider == "github":
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            headers = {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            payload = {
                "title": title,
                "head": state.target_branch,
                "base": state.base_branch,
                "body": body,
            }
        # Add elif for gitlab here
        else:
            logger.error(f"VCS provider '{self.config.vcs_provider}' not supported for PR creation.")
            return None

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)
            pr_data = response.json()
            pr_url = pr_data.get("html_url")
            if pr_url:
                state.add_log(f"Pull Request created successfully: {pr_url}")
                state.pr_url = pr_url
                return pr_url
            else:
                logger.error(f"PR created, but URL not found in response: {pr_data}")
                raise VCSError("PR URL not found in API response.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create Pull Request: {e}")
            logger.error(f"Response body: {e.response.text if e.response else 'No response'}")
            raise VCSError(f"Failed to create PR via API: {e}")


class LLMService:
    """Wrapper for interacting with the Large Language Model."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        if self.config.gemini_api_key:
             try:
                self.model = genai.GenerativeModel(self.config.llm_model_name)
                logger.info(f"LLM model '{self.config.llm_model_name}' initialized.")
             except Exception as e:
                 logger.error(f"Failed to initialize LLM model: {e}")
                 self.model = None # Ensure model is None if init fails
        else:
            logger.warning("LLM Service initialized without API key. LLM calls will fail.")

    def _call_llm(self, prompt: str, is_json_output: bool = False) -> str:
        """Makes a call to the configured LLM."""
        if not self.model:
            raise LLMError("LLM model not initialized or API key missing.")

        logger.info(f"Calling LLM (model: {self.config.llm_model_name})...")
        # logger.debug(f"LLM Prompt:\n---PROMPT START---\n{prompt}\n---PROMPT END---")

        generation_config = genai.types.GenerationConfig(
            # Adjust temperature for creativity vs predictability
            temperature=0.4,
            # Max output tokens - adjust based on expected output size
            max_output_tokens=8192,
        )
        if is_json_output:
             generation_config.response_mime_type="application/json"


        try:
            response = self.model.generate_content(prompt, generation_config=generation_config)
            # Accessing the text directly assumes the response structure
            # Handle potential errors or different response structures if needed
            if hasattr(response, 'text'):
                result_text = response.text
                logger.info("LLM call successful.")
                # logger.debug(f"LLM Response:\n---RESPONSE START---\n{result_text}\n---RESPONSE END---")
                return result_text
            elif hasattr(response, 'parts'):
                 # Handle cases where the response might be structured differently (e.g., blocked prompts)
                 result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                 if result_text:
                     logger.info("LLM call successful (parsed from parts).")
                     # logger.debug(f"LLM Response:\n---RESPONSE START---\n{result_text}\n---RESPONSE END---")
                     return result_text
                 else: # Check for safety ratings etc.
                    logger.warning(f"LLM response might be empty or blocked. Full response: {response}")
                    raise LLMError(f"LLM response is empty or potentially blocked. Reason: {response.prompt_feedback}")

            else:
                logger.error(f"Unexpected LLM response structure: {response}")
                raise LLMError("Unexpected LLM response structure.")

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Catch specific API errors if the library provides them
            raise LLMError(f"LLM API call failed: {e}")


    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Uses LLM to parse the task description."""
        prompt = f"""
Analyze the following software development task description. Extract the key requirements,
the main goal, affected areas (e.g., UI, API, Database, specific components/modules if mentioned),
and any constraints or non-functional requirements.

Task Description:
"{task_description}"

Output the analysis as a JSON object with the following keys:
- "goal": A concise summary of the main objective (string).
- "requirements": A list of specific functional requirements derived from the task (list of strings).
- "affected_areas": A list of general areas or specific components likely impacted (list of strings).
- "constraints": Any mentioned constraints or non-functional requirements (list of strings).
- "clarifications_needed": A list of questions if the task is ambiguous or requires more information (list of strings).
"""
        try:
            response_text = self._call_llm(prompt, is_json_output=True)
            # Basic cleaning of potential markdown backticks
            cleaned_response = response_text.strip().strip('```json').strip('```').strip()
            analysis = json.loads(cleaned_response)
            # Validate expected keys?
            return analysis
        except (json.JSONDecodeError, LLMError) as e:
            logger.error(f"Failed to analyze task description: {e}")
            # Return a default structure or re-raise
            return {
                 "goal": f"Failed to parse: {task_description[:100]}...",
                 "requirements": [], "affected_areas": ["Unknown"],
                 "constraints": [], "clarifications_needed": ["LLM analysis failed."]
             }

    def identify_files_to_change(self, state: TaskState, file_tree: str) -> List[str]:
        """Uses LLM to identify files relevant to the task."""
        prompt = f"""
Given the following task description and project file structure, identify the files
that are most likely to require modification or inspection to implement the task.
Focus on the core logic, related components, and potentially tests.

Task Description:
{state.task_description}

Project File Structure:
{file_tree}

List the full paths of the relevant files within the project structure, one file per line.
Do not include directories, only file paths. If unsure, list the most probable candidates.
Provide only the list of file paths, no explanation.
"""
        response = self._call_llm(prompt)
        files = [line.strip() for line in response.splitlines() if line.strip() and '.' in line.strip()] # Basic filtering
        logger.info(f"LLM identified potential files: {files}")
        return files

    def identify_coding_patterns(self, state: TaskState, relevant_code_samples: Dict[str, str]) -> str:
         """Uses LLM to identify coding patterns in relevant code snippets."""
         code_snippets_text = "\n\n".join([f"--- File: {path} ---\n{code[:1000]}..." for path, code in relevant_code_samples.items()]) # Limit context size
         if not code_snippets_text:
              return "No relevant code samples provided to identify patterns."

         prompt = f"""
Analyze the provided code snippets from a project to identify key coding patterns,
style conventions, and common practices used in this codebase. Consider aspects like:
- Naming conventions (variables, functions, classes)
- Error handling strategy (try-catch, error codes, specific libraries)
- State management patterns (if applicable, e.g., Redux, Vuex, context API, global variables)
- API call patterns (e.g., fetch, axios, specific wrappers)
- Testing patterns (e.g., libraries used, structure of test files, mocking)
- Code formatting and style (indentation, comments, etc.)

Task Description (for context):
{state.task_description}

Code Snippets:
{code_snippets_text}

Summarize the observed patterns and conventions concisely. This information will be used
to ensure generated code matches the existing style.
"""
         patterns = self._call_llm(prompt)
         logger.info(f"LLM identified coding patterns summary: {patterns[:200]}...")
         return patterns

    def generate_code_patch(self, state: TaskState, relevant_files_content: Dict[str, str]) -> str:
        """Uses LLM to generate code changes as a patch."""
        file_content_text = "\n\n".join([f"--- File: {path} ---\n```\n{content}\n```" for path, content in relevant_files_content.items()])
        if not file_content_text:
             raise LLMError("No relevant file content provided for patch generation.")

        prompt = f"""
You are an AI assistant tasked with modifying a codebase based on a given task.
Generate the necessary code changes to implement the task described below.
Adhere strictly to the identified coding patterns and style of the existing codebase.

Task Description:
{state.task_description}

Identified Coding Patterns & Style:
{state.coding_patterns}

Relevant Files and their Current Content:
{file_content_text}

Instructions:
1. Analyze the task and the provided code.
2. Determine the exact changes needed (additions, modifications, deletions).
3. Output the changes ONLY in the unified diff format (a git patch).
4. Ensure the patch applies cleanly to the provided file content.
5. Make sure to include necessary imports or declarations.
6. If new files are needed, represent their creation in the diff format (diff --git a/new/file/path b/new/file/path, index 0000000..xxxxxxx, --- /dev/null, +++ b/new/file/path, @@ -0,0 +1,N @@, + <content>).
7. Generate ONLY the diff content, without any explanation before or after. Start directly with 'diff --git ...' or the first hunk '--- a/...' '+++ b/...'.

Generate the patch now:
"""
        patch_content = self._call_llm(prompt)
        # Basic validation: check if it looks like a diff
        if not patch_content.strip().startswith(("diff --git", "--- a/", "+++ b/")):
            logger.warning(f"LLM output doesn't look like a standard diff:\n{patch_content[:300]}...")
            # Decide: raise error or try anyway? Let's try, apply_patch might handle it.
            # raise LLMError("Generated output does not appear to be a valid diff patch.")
        else:
            logger.info(f"LLM generated patch:\n{patch_content[:500]}...") # Log beginning of patch
        return patch_content.strip() # Remove leading/trailing whitespace

    def analyze_error(self, state: TaskState, error_output: str, code_context: Dict[str, str]) -> str:
        """Uses LLM to analyze build/lint/test errors and suggest cause."""
        file_content_text = "\n\n".join([f"--- File: {path} ---\n```\n{content}\n```" for path, content in code_context.items()])

        prompt = f"""
An error occurred during the verification phase (build, lint, or test) after applying code changes.
Analyze the error output and the relevant code context to determine the likely cause of the error.

Task Description:
{state.task_description}

Error Output:


{error_output}

Relevant Code Context (files involved in the last patch or error message):
{file_content_text}

Instructions:
1. Identify the specific error message(s).
2. Relate the error to the code changes or context.
3. Briefly explain the most likely root cause of the error.
4. Provide only the explanation of the cause. Do not suggest a fix yet.
"""
        analysis = self._call_llm(prompt)
        logger.info(f"LLM error analysis: {analysis}")
        return analysis

    def generate_fix_patch(self, state: TaskState, error_output: str, error_analysis: str, code_context: Dict[str, str]) -> str:
        """Uses LLM to generate a patch to fix the identified error."""
        file_content_text = "\n\n".join([f"--- File: {path} ---\n```\n{content}\n```" for path, content in code_context.items()])

        prompt = f"""
You are an AI assistant fixing a bug in the code. An error occurred, and an analysis suggests the cause.
Generate a patch (in unified diff format) to fix the error based on the analysis and context.
Adhere to the project's coding patterns.

Task Description:
{state.task_description}

Original Error Output:
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

{error_output}

Error Cause Analysis:
{error_analysis}

Relevant Code Context (potentially containing the error):
{file_content_text}

Coding Patterns & Style:
{state.coding_patterns}

Instructions:
1. Understand the error, the analysis, and the code.
2. Generate the minimal changes needed to fix the error.
3. Output the changes ONLY in the unified diff format (a git patch).
4. Ensure the patch applies cleanly to the provided code context.
5. Generate ONLY the diff content, without any explanation. Start directly with 'diff --git ...' or '--- a/...'.

Generate the fix patch now:
"""
        fix_patch = self._call_llm(prompt)
         # Basic validation
        if not fix_patch.strip().startswith(("diff --git", "--- a/", "+++ b/")):
            logger.warning(f"LLM fix output doesn't look like a standard diff:\n{fix_patch[:300]}...")
            # raise LLMError("Generated fix output does not appear to be a valid diff patch.")
        else:
             logger.info(f"LLM generated fix patch:\n{fix_patch[:500]}...")
        return fix_patch.strip()

    def generate_tests(self, state: TaskState, changed_files_content: Dict[str, str], existing_tests_samples: Dict[str, str]) -> str:
        """Uses LLM to generate new tests for the changes."""
        changed_code_text = "\n\n".join([f"--- File: {path} ---\n```\n{content}\n```" for path, content in changed_files_content.items()])
        existing_tests_text = "\n\n".join([f"--- Test File: {path} ---\n```\n{content[:1000]}...\n```" for path, content in existing_tests_samples.items()]) # Limit context

        prompt = f"""
You are an AI assistant writing tests for new or modified code.
Generate relevant unit tests (and potentially integration tests if simple) for the changes described.
Follow the testing patterns observed in the existing tests provided.

Task Description:
{state.task_description}

Code Changes Implemented (New/Modified Files):
{changed_code_text}

Existing Test Samples (for pattern reference):
{existing_tests_text if existing_tests_text else "No existing test samples provided."}

Project Coding Patterns & Style:
{state.coding_patterns}

Instructions:
1. Analyze the code changes and the task.
2. Identify key functionalities or logic paths to test.
3. Based on existing test patterns (framework, structure, assertion style), write new test cases.
4. If existing test files are relevant, provide changes as a patch (unified diff format).
5. If new test files are needed, provide their full content using the diff format for new files (starting with `diff --git a/path/to/new_test.spec.js b/path/to/new_test.spec.js`).
6. Aim for reasonable coverage of the new/modified logic.
7. Generate ONLY the test code or patch content, without explanation. If generating a patch, start directly with 'diff --git ...' or '--- a/...'.

Generate the tests or test patch now:
"""
        test_patch_or_code = self._call_llm(prompt)
        logger.info(f"LLM generated tests (or patch):\n{test_patch_or_code[:500]}...")
        # We assume this output is also a patch/diff for consistency
        # If it generates full file content, the apply logic needs adjustment
        return test_patch_or_code.strip()

    def generate_commit_message(self, state: TaskState) -> str:
        """Uses LLM to generate a commit message."""
        # Ideally, provide diff summary or list of changed files to LLM
        # For simplicity now, just use task description
        prompt = f"""
Generate a concise and informative Git commit message summarizing the changes made for the following task.
Follow conventional commit standards if possible (e.g., "feat: add user login functionality").

Task Description:
{state.task_description}

Generate only the commit message title (and optionally body, separated by a blank line).
"""
        message = self._call_llm(prompt)
        logger.info(f"LLM generated commit message: {message}")
        # Simple cleanup: take first line as title, rest as body if present
        lines = message.strip().split('\n')
        if len(lines) > 1 and not lines[1].strip(): # Check for blank line separation
             return f"{lines[0].strip()}\n\n{' '.join(l.strip() for l in lines[2:])}"
        else:
             return lines[0].strip()


    def generate_pr_description(self, state: TaskState) -> str:
        """Uses LLM to generate a Pull Request description."""
        # Ideally provide summary of changes, link to ticket etc.
        prompt = f"""
Generate a Pull Request description for the changes made to address the following task.
The PR was automatically generated by the CodeCraft AI agent.

Task Description:
{state.task_description}

Changes Made Summary:
(CodeCraft AI Agent Note: Include a brief summary of key changes here if available, otherwise LLM should infer from task)
- Implemented feature X / Refactored module Y / Fixed bug Z
- Added/modified files: {', '.join(state.relevant_files) if state.relevant_files else 'N/A'}
- Added unit tests for the new functionality.

Verification Status:
- Build: Passed
- Linter: Passed
- Tests: Passed

Instructions:
- Briefly restate the original task goal.
- Summarize the key changes made.
- Mention that automated checks (build, lint, tests) have passed.
- Include a call to action for human review.
- Format using Markdown.
"""
        description = self._call_llm(prompt)
        logger.info(f"LLM generated PR description: {description[:200]}...")
        return description.strip()


class CodeAnalyzer:
    """Analyzes the codebase structure, dependencies, and patterns."""

    def __init__(self, config: Config, sandbox_manager: SandboxManager, llm_service: LLMService):
        self.config = config
        self.sandbox_manager = sandbox_manager
        self.llm_service = llm_service

    def _list_files(self, state: TaskState, directory: str = ".") -> str:
        """Lists files in a directory within the sandbox, returns as a string."""
        # Exclude common noise like node_modules, .git, virtualenvs etc.
        # Using find is more flexible than ls
        exclude_dirs = ["./.git", "./node_modules", "./venv", "./.venv", "./dist", "./build"]
        exclude_paths_str = " ".join([f"-path '{d}' -prune -o" for d in exclude_dirs])
        command = f"find {directory} {exclude_paths_str} -type f -print"

        exit_code, output = self.sandbox_manager.run_command(state, command, workdir=state.repo_path_in_sandbox)
        if exit_code != 0:
            logger.warning(f"Failed to list files in sandbox: {output}")
            return "Error listing files."
        return output

    def _read_file_content(self, state: TaskState, file_path: str) -> Optional[str]:
        """Reads the content of a specific file from the sandbox."""
         # Basic check to prevent reading huge files entirely
        check_size_cmd = f"stat -c %s {file_path}"
        exit_code_size, out_size = self.sandbox_manager.run_command(state, check_size_cmd, workdir=state.repo_path_in_sandbox)
        try:
             if exit_code_size == 0 and int(out_size.strip()) > 1_000_000: # 1MB limit
                  logger.warning(f"File '{file_path}' is larger than 1MB, skipping read.")
                  return f"Error: File '{file_path}' is too large (>1MB)."
        except ValueError:
             logger.warning(f"Could not parse file size for '{file_path}': {out_size}")
             # Proceed cautiously

        # Use 'cat' to read the file content
        read_command = f"cat {file_path}"
        exit_code, output = self.sandbox_manager.run_command(state, read_command, workdir=state.repo_path_in_sandbox)
        if exit_code == 0:
            # Replace potential carriage returns for consistency
            return output.replace('\r\n', '\n').replace('\r', '\n')
        else:
            logger.warning(f"Failed to read file '{file_path}' in sandbox. Output: {output}")
            return None # Indicate failure to read

    def detect_project_type_and_commands(self, state: TaskState):
        """Detects project type and common commands (install, build, test, lint)."""
        state.add_log("Detecting project type and commands...")
        # Default commands initially
        state.install_commands = list(self.config.default_install_commands)
        state.build_commands = list(self.config.default_build_commands)
        state.lint_commands = list(self.config.default_lint_commands)
        state.test_commands = list(self.config.default_test_commands)

        # --- Try detecting based on files ---
        detected_type = "Unknown"
        project_files = self._list_files(state).splitlines()

        # Python checks
        if any(f.endswith("requirements.txt") for f in project_files):
            detected_type = "Python (requirements.txt)"
            state.install_commands = ["pip install -r requirements.txt"]
            state.test_commands = ["pytest"] # Common default
            state.lint_commands = ["flake8 .", "pylint ."] # Common defaults
            # Check for Flask/Django structure? (manage.py, app.py)
            if any(f.endswith("manage.py") for f in project_files):
                 detected_type = "Python (Django)"
                 state.test_commands = ["python manage.py test"]
            elif any(f.endswith("app.py") or f.endswith("wsgi.py") for f in project_files):
                 detected_type = "Python (Flask/WSGI)"

        elif any(f.endswith("pyproject.toml") for f in project_files):
            detected_type = "Python (pyproject.toml)"
            state.install_commands = ["pip install ."] # Basic install
             # Check for specific tools like poetry, pdm? Needs file content parsing
            pyproject_content = self._read_file_content(state, "./pyproject.toml")
            if pyproject_content:
                if "[tool.poetry]" in pyproject_content:
                     detected_type = "Python (Poetry)"
                     state.install_commands = ["poetry install"]
                     state.test_commands = ["poetry run pytest"]
                     state.lint_commands = ["poetry run flake8", "poetry run pylint"]
                # Add checks for PDM, etc.

        # Node.js check
        elif any(f.endswith("package.json") for f in project_files):
            detected_type = "Node.js"
            package_json_content = self._read_file_content(state, "./package.json")
            install_cmd = "npm install"
            if any(f.endswith("yarn.lock") for f in project_files):
                install_cmd = "yarn install"
            elif any(f.endswith("pnpm-lock.yaml") for f in project_files):
                 install_cmd = "pnpm install"
            state.install_commands = [install_cmd]

            # Try parsing scripts from package.json
            if package_json_content:
                try:
                    package_data = json.loads(package_json_content)
                    scripts = package_data.get("scripts", {})
                    # Look for common script names
                    state.build_commands = [f"{install_cmd.split()[0]} run build"] if "build" in scripts else []
                    state.test_commands = [f"{install_cmd.split()[0]} test"] if "test" in scripts else []
                    state.lint_commands = [f"{install_cmd.split()[0]} run lint"] if "lint" in scripts else []
                except json.JSONDecodeError:
                    logger.warning("Failed to parse package.json")
            # Fallback if scripts not found/parsed
            if not state.test_commands: state.test_commands = ["npm test"] # Default Node test
            if not state.lint_commands: state.lint_commands = ["npm run lint"] # Default Node lint

        # Add more detection logic here (Java/Maven/Gradle, Ruby/Bundler, Go, etc.)

        # --- Final Logging ---
        state.project_type = detected_type
        state.add_log(f"Detected Project Type: {state.project_type}")
        state.add_log(f"Using Install Commands: {state.install_commands}")
        state.add_log(f"Using Build Commands: {state.build_commands}")
        state.add_log(f"Using Lint Commands: {state.lint_commands}")
        state.add_log(f"Using Test Commands: {state.test_commands}")
        # TODO: Allow overriding these commands via Agent config per project if detection fails.


    def find_relevant_files_and_patterns(self, state: TaskState):
        """Identifies relevant files and coding patterns using LLM."""
        state.status = "analyzing"
        state.add_log("Analyzing codebase to find relevant files and patterns...")

        # 1. Get file tree
        file_tree = self._list_files(state)
        if not file_tree or "Error listing files" in file_tree:
            raise CodeCraftError("Could not get file listing from sandbox.")

        # 2. Ask LLM to identify relevant files
        try:
            state.relevant_files = self.llm_service.identify_files_to_change(state, file_tree)
            if not state.relevant_files:
                 # Fallback: Maybe just list all non-ignored files? Or fail?
                 logger.warning("LLM did not identify any specific relevant files.")
                 # Let's try to proceed, maybe the generate step can work without specifics? Risky.
                 # Or maybe try asking LLM again with a simpler prompt?
                 # For now, we will proceed but log a strong warning.
                 state.add_log("Warning: LLM failed to identify relevant files. Generation might be inaccurate.")
                 # Optional: Get _all_ files as fallback?
                 # state.relevant_files = [f for f in file_tree.splitlines() if f.strip()]

        except LLMError as e:
            logger.error(f"LLM failed during file identification: {e}")
            # Handle error - maybe try a simpler analysis or fail the task
            raise CodeCraftError(f"LLM error during code analysis: {e}")

        # 3. Read content of relevant files
        relevant_files_content: Dict[str, str] = {}
        files_to_read = state.relevant_files[:10] # Limit number of files to read for performance/context size
        state.add_log(f"Reading content of up to {len(files_to_read)} relevant files: {files_to_read}...")
        for file_path in files_to_read:
            # Normalize path relative to repo root if needed
            abs_file_path = file_path.strip() # Assume paths from find are relative or absolute already
            content = self._read_file_content(state, abs_file_path)
            if content is not None:
                relevant_files_content[abs_file_path] = content
            else:
                 state.add_log(f"Could not read content for: {abs_file_path}")

        if not relevant_files_content:
             # This might happen if relevant_files were empty or reading failed
             logger.warning("No content read from relevant files. Pattern analysis might be inaccurate.")
             # Proceed without pattern analysis, or maybe analyze some common files?
             state.coding_patterns = "Could not read relevant files to determine specific patterns."
        else:
            # 4. Ask LLM to identify patterns from samples
            try:
                state.coding_patterns = self.llm_service.identify_coding_patterns(state, relevant_files_content)
            except LLMError as e:
                logger.error(f"LLM failed during pattern identification: {e}")
                state.coding_patterns = f"LLM error during pattern analysis: {e}" # Store error as pattern info

        state.add_log("Code analysis phase completed.")

    def get_code_context(self, state: TaskState, file_paths: List[str]) -> Dict[str, str]:
         """Reads content of specified files to provide context."""
         context: Dict[str, str] = {}
         state.add_log(f"Reading context for files: {file_paths}")
         for file_path in file_paths[:15]: # Limit context files again
              content = self._read_file_content(state, file_path.strip())
              if content:
                   context[file_path.strip()] = content
              else:
                   context[file_path.strip()] = f"Error: Could not read file '{file_path}'."
         return context


class CodeGenerator:
    """Generates code patches using the LLM."""

    def __init__(self, config: Config, llm_service: LLMService, code_analyzer: CodeAnalyzer):
        self.config = config
        self.llm_service = llm_service
        self.code_analyzer = code_analyzer

    def generate_initial_patch(self, state: TaskState) -> Optional[str]:
        """Generates the first code patch based on the task."""
        state.status = "generating"
        state.add_log("Generating initial code patch...")

        if not state.relevant_files:
            state.add_log("No relevant files identified, attempting generation without specific context.")
            # This is less likely to succeed but fulfills the "no TODO" requirement
            # LLM might hallucinate file paths or make very generic changes.
            relevant_files_content = {}
        else:
            # Get content of relevant files needed for generation context
            relevant_files_content = self.code_analyzer.get_code_context(state, state.relevant_files)

        if not relevant_files_content and state.relevant_files:
             state.add_log("Warning: Failed to read content of relevant files. Patch generation might fail or be inaccurate.")
             # Proceed, but quality is compromised.

        try:
            patch = self.llm_service.generate_code_patch(state, relevant_files_content)
            state.current_patch = patch
            return patch
        except LLMError as e:
            logger.error(f"LLM failed during initial patch generation: {e}")
            state.last_error_output = f"LLM error during initial patch generation: {e}"
            return None # Indicate failure

class VerificationRunner:
    """Runs build, lint, and test commands in the sandbox."""

    def __init__(self, config: Config, sandbox_manager: SandboxManager):
        self.config = config
        self.sandbox_manager = sandbox_manager

    def _run_verification_commands(self, state: TaskState, commands: List[str], step_name: str) -> Tuple[bool, str]:
        """Helper to run a list of commands for a verification step."""
        if not commands:
            state.add_log(f"No {step_name} commands configured/detected, skipping step.")
            return True, f"{step_name} skipped (no commands)."

        state.add_log(f"Running {step_name} step...")
        full_output = ""
        for command in commands:
            exit_code, output = self.sandbox_manager.run_command(state, command)
            full_output += f"\n--- Command: {command} ---\n{output}"
            if exit_code != 0:
                state.add_log(f"{step_name} command '{command}' failed.")
                return False, full_output.strip() # Failed
        state.add_log(f"{step_name} step completed successfully.")
        return True, full_output.strip() # Success

    def run_build(self, state: TaskState) -> Tuple[bool, str]:
        """Runs build commands."""
        return self._run_verification_commands(state, state.build_commands, "Build")

    def run_lint(self, state: TaskState) -> Tuple[bool, str]:
        """Runs static analysis/lint commands."""
        return self._run_verification_commands(state, state.lint_commands, "Lint")

    def run_tests(self, state: TaskState) -> Tuple[bool, str]:
        """Runs test commands."""
        return self._run_verification_commands(state, state.test_commands, "Test")

class DebuggerAssistant:
    """Handles the debugging loop using LLM."""

    def __init__(self, config: Config, llm_service: LLMService, code_analyzer: CodeAnalyzer, vcs_manager: VCSManager):
        self.config = config
        self.llm_service = llm_service
        self.code_analyzer = code_analyzer
        self.vcs_manager = vcs_manager # Needed to revert failed patches

    def debug_step(self, state: TaskState, error_output: str) -> Optional[str]:
        """Performs one iteration of analyzing error and generating a fix."""
        state.status = "debugging"
        state.debug_iteration += 1
        state.add_log(f"Debugging attempt {state.debug_iteration}/{self.config.max_debug_iterations}...")
        state.last_error_output = error_output

        # 1. Revert the failed patch first to get back to previous state
        # This is tricky. A better approach might be to just work from the broken state,
        # but reverting is simpler to conceptualize here. Requires git in sandbox.
        # If patch apply failed partially (with .rej files), this might be complex.
        # Let's assume for now the patch applied fully but caused runtime/test errors.
        # If `git apply` itself failed, we might already be in a clean state.
        # A simple approach: `git reset --hard HEAD` inside the sandbox.
        state.add_log("Reverting failed changes before generating fix...")
        try:
             self.vcs_manager._run_git_command(state, "reset --hard HEAD")
             state.add_log("Reverted changes to HEAD.")
        except VCSError as e:
             logger.error(f"Failed to revert changes during debug: {e}. Attempting to proceed anyway.")
             state.add_log(f"Warning: Failed to revert changes: {e}. State might be inconsistent.")
             # Proceeding might generate a patch based on a broken state.


        # 2. Get code context around the error
        # How to determine relevant files? Use original relevant_files + potentially files mentioned in error?
        # Simple approach: use state.relevant_files again.
        # More advanced: parse error_output for file paths.
        error_files = re.findall(r'[\./\w-]+\.(?:py|js|jsx|ts|tsx|java|go|rb|php)\b', error_output) # Basic file regex
        context_files = list(set(state.relevant_files + error_files))
        code_context = self.code_analyzer.get_code_context(state, context_files)

        if not code_context:
            state.add_log("Could not get code context for debugging. Cannot generate fix.")
            return None

        # 3. Analyze the error with LLM
        try:
            error_analysis = self.llm_service.analyze_error(state, error_output, code_context)
        except LLMError as e:
            logger.error(f"LLM failed during error analysis: {e}")
            state.last_error_output += f"\nLLM error during analysis: {e}"
            return None # Cannot proceed without analysis

        # 4. Generate a fix patch with LLM
        try:
            fix_patch = self.llm_service.generate_fix_patch(state, error_output, error_analysis, code_context)
            state.current_patch = fix_patch
            return fix_patch
        except LLMError as e:
            logger.error(f"LLM failed during fix patch generation: {e}")
            state.last_error_output += f"\nLLM error during fix generation: {e}"
            return None # Failed to generate fix


class CodeCraftAI:
    """Orchestrator for the AI coding agent."""

    def __init__(self):
        self.config = CONFIG
        self.sandbox_manager = SandboxManager(self.config)
        self.vcs_manager = VCSManager(self.config, self.sandbox_manager)
        self.llm_service = LLMService(self.config)
        self.code_analyzer = CodeAnalyzer(self.config, self.sandbox_manager, self.llm_service)
        self.code_generator = CodeGenerator(self.config, self.llm_service, self.code_analyzer)
        self.verifier = VerificationRunner(self.config, self.sandbox_manager)
        self.debugger = DebuggerAssistant(self.config, self.llm_service, self.code_analyzer, self.vcs_manager)
        self.state: Optional[TaskState] = None

    def _run_verification_cycle(self, state: TaskState) -> bool:
        """Runs build, lint, test and returns True if all pass."""
        state.status = "verifying"
        state.add_log("Starting verification cycle...")

        # 1. Build check
        build_ok, build_output = self.verifier.run_build(state)
        if not build_ok:
            state.add_log("Build failed.")
            state.last_error_output = build_output
            return False

        # 2. Lint check
        lint_ok, lint_output = self.verifier.run_lint(state)
        if not lint_ok:
            state.add_log("Linting failed.")
            state.last_error_output = lint_output
            return False # Treat lint errors as blocking

        # 3. Test check
        test_ok, test_output = self.verifier.run_tests(state)
        if not test_ok:
            state.add_log("Tests failed.")
            state.last_error_output = test_output
            return False

        state.add_log("Verification cycle passed successfully.")
        state.last_error_output = None # Clear last error on success
        return True

    def run(self, task_description: str, repo_url: str, base_branch: str) -> Optional[str]:
        """Executes the end-to-end process for a given task."""
        self.state = TaskState(
            task_description=task_description,
            repo_url=repo_url,
            base_branch=base_branch
        )
        state = self.state # Local alias for convenience

        try:
            # --- 1. Setup Phase ---
            state.add_log("Starting CodeCraft AI task...")
            state.status = "initializing"
            # 1.1 Create Sandbox
            self.sandbox_manager.create_sandbox(state)
            # 1.2 Clone Repo
            self.vcs_manager.clone_repo(state)
            # 1.3 Create Branch
            self.vcs_manager.create_branch(state)
            # 1.4 Detect Project & Install Deps
            self.code_analyzer.detect_project_type_and_commands(state)
            install_ok, install_output = self.verifier._run_verification_commands(state, state.install_commands, "Install Dependencies")
            if not install_ok:
                raise CodeCraftError(f"Failed to install project dependencies:\n{install_output}")
            state.add_log("Environment setup complete.")

            # --- 2. Analysis Phase ---
            # 2.1 Analyze Task (Optional - can be integrated into prompts)
            # task_analysis = self.llm_service.analyze_task(state.task_description)
            # state.add_log(f"Task analysis: {task_analysis}")
            # if task_analysis.get("clarifications_needed"):
            #    raise CodeCraftError(f"Task is unclear: {task_analysis['clarifications_needed']}")

            # 2.2 Analyze Codebase (Find files, patterns)
            self.code_analyzer.find_relevant_files_and_patterns(state)

            # --- 3. Generation & Verification Loop ---
            state.add_log("Entering Generation & Verification Loop...")
            success = False
            while state.debug_iteration < self.config.max_debug_iterations:
                # 3.1 Generate Patch (Initial or Fix)
                if state.debug_iteration == 0:
                    patch_content = self.code_generator.generate_initial_patch(state)
                else: # In debug loop
                    if state.last_error_output is None:
                        # Should not happen if loop condition is correct, but defensive check
                        state.add_log("Error: Debug loop entered without error output. Aborting.")
                        break
                    patch_content = self.debugger.debug_step(state, state.last_error_output)

                if patch_content is None:
                    state.add_log(f"Failed to generate patch (attempt {state.debug_iteration + 1}). Check logs for LLM errors.")
                    # If initial generation fails, break. If fix generation fails, break.
                    break

                # 3.2 Apply Patch
                patch_applied = self.vcs_manager.apply_patch(state, patch_content)
                if not patch_applied:
                    state.add_log("Failed to apply the generated patch cleanly.")
                    # Error stored in state.last_error_output by apply_patch
                    # Continue to next debug iteration to fix the patch application issue
                    state.status = "debugging" # Ensure status reflects we need debug
                    continue # Skip verification, go straight to debug

                # 3.3 Verify Changes
                state.add_log(f"Running verification after applying patch (Attempt {state.debug_iteration + 1})...")
                if self._run_verification_cycle(state):
                    state.add_log("Patch verified successfully!")
                    success = True
                    break # Exit the loop on success
                else:
                    state.add_log(f"Verification failed (Attempt {state.debug_iteration + 1}). Entering debug step.")
                    # Error is already in state.last_error_output from _run_verification_cycle
                    # Loop will continue to debugger.debug_step

            # --- 4. Post-Loop Check ---
            if not success:
                state.status = "failed"
                error_msg = f"Failed to achieve passing state after {state.debug_iteration + 1} attempts. Last error:\n{state.last_error_output or 'Unknown verification error.'}"
                state.add_log(error_msg)
                raise CodeCraftError(error_msg)

            state.add_log("Main functionality implemented and verified.")

            # --- 5. Generate Tests ---
            state.status = "testing"
            state.add_log("Attempting to generate tests...")
            # Get final content of changed files
            # Need to figure out which files were *actually* changed by the final successful patch
            # We could run `git diff --name-only HEAD~1` ?
            try:
                diff_output = self.vcs_manager._run_git_command(state, "diff --name-only HEAD~1 HEAD")
                changed_files = diff_output.strip().splitlines()
                state.add_log(f"Files changed in the last commit: {changed_files}")
            except VCSError as e:
                 logger.warning(f"Could not determine changed files from git diff: {e}. Using initially relevant files for test generation context.")
                 changed_files = state.relevant_files # Fallback

            changed_content = self.code_analyzer.get_code_context(state, changed_files)
            # Find existing tests (simple approach: look for files matching pattern)
            # Use find command again?
            # find . -path '*/test/*' -o -path '*/tests/*' -o -name '*.test.*' -o -name '*.spec.*'
            find_tests_cmd = "find . -path '*/test/*' -prune -o -path '*/tests/*' -prune -o -name '*[._]test.*' -print -o -name '*[._]spec.*' -print"
            test_find_exit, test_find_out = self.sandbox_manager.run_command(state, find_tests_cmd, workdir=state.repo_path_in_sandbox)
            existing_test_files = []
            if test_find_exit == 0:
                 existing_test_files = test_find_out.strip().splitlines()
                 state.add_log(f"Found potential existing test files: {existing_test_files[:5]}")

            existing_tests_content = self.code_analyzer.get_code_context(state, existing_test_files[:5]) # Limit context

            try:
                test_patch = self.llm_service.generate_tests(state, changed_content, existing_tests_content)
                if test_patch:
                    state.add_log("Applying generated tests patch...")
                    test_patch_applied = self.vcs_manager.apply_patch(state, test_patch)
                    if test_patch_applied:
                         state.add_log("Generated tests applied. Re-running test verification...")
                         # Re-run only tests after adding new tests
                         test_ok, test_output = self.verifier.run_tests(state)
                         if not test_ok:
                              # Tests failed after adding generated tests. Log it but proceed to PR.
                              state.add_log("Warning: Tests failed after adding LLM-generated tests. Manual review needed.")
                              # Optionally: Could try a single debug step for tests here.
                              state.last_error_output = f"Tests failed after adding generated tests:\n{test_output}"
                         else:
                              state.add_log("Tests passed with newly generated tests included.")
                    else:
                         state.add_log("Warning: Failed to apply generated tests patch.")
                         # Proceed without the generated tests
                else:
                    state.add_log("LLM did not generate any tests.")

            except LLMError as e:
                logger.error(f"LLM failed during test generation: {e}")
                state.add_log("Warning: LLM error during test generation.")

            # --- 6. Commit and Create PR ---
            state.status = "committing"
            state.add_log("Generating commit message...")
            commit_message = self.llm_service.generate_commit_message(state)

            # Commit changes (including applied tests)
            self.vcs_manager.commit_changes(state, commit_message)

            # Push branch
            self.vcs_manager.push_branch(state)

            # Create PR
            state.add_log("Generating PR description...")
            pr_title = commit_message.split('\n')[0] # Use first line of commit message as title
            pr_body = self.llm_service.generate_pr_description(state)
            pr_url = self.vcs_manager.create_pull_request(state, pr_title, pr_body)

            if pr_url:
                state.status = "completed"
                state.add_log(f"Task completed successfully! Pull Request created: {pr_url}")
                return pr_url
            else:
                 state.status = "failed" # Consider it failed if PR creation fails
                 state.add_log("Task completed, changes pushed, but failed to create Pull Request.")
                 raise CodeCraftError("Changes pushed, but failed to create Pull Request.")

        except CodeCraftError as e:
            state.status = "failed"
            logger.error(f"CodeCraft AI task failed: {e}")
            state.add_log(f"Error: {e}")
            # Add final error output to state log if available
            if state.last_error_output and str(e) not in state.last_error_output:
                 state.add_log(f"Underlying error output:\n{state.last_error_output}")

            return None # Indicate failure
        except Exception as e:
            # Catch unexpected errors
            state.status = "failed"
            logger.exception("An unexpected error occurred during CodeCraft AI execution.")
            state.add_log(f"Unexpected critical error: {e}")
            return None # Indicate failure
        finally:
            # --- Cleanup ---
            if state and state.container:
                self.sandbox_manager.cleanup_sandbox(state)
            # Save logs?
            log_filename = f"codecraft_log_{uuid.uuid4().hex[:8]}.log"
            with open(log_filename, "w") as f:
                 if state:
                     json.dump(state.log, f, indent=2)
                 else:
                     f.write("Failed before state initialization.")
            logger.info(f"Execution log saved to {log_filename}")


# --- Main Execution ---

if __name__ == "__main__":
    print("CodeCraft AI Agent - Proof of Concept")
    print("-" * 30)

    if not CONFIG.gemini_api_key:
         print("Error: GEMINI_API_KEY environment variable not set.")
         sys.exit(1)
    if not CONFIG.github_token:
         print("Warning: GITHUB_TOKEN environment variable not set. PR creation will likely fail for private repos or API interaction.")
         # Decide if this is critical - maybe allow proceeding without PR? For now, just warn.

    # --- Example Usage ---
    # Replace with actual task and repo details
    # Ensure the repo is accessible (public or with token)
    # Ensure the base branch exists
    example_task = "Add a new API endpoint `/hello` to the Flask application that returns a JSON object `{'message': 'Hello, Coder!'}`. Include a basic pytest test case for this endpoint."
    # example_repo = "https://github.com/your-username/your-flask-project.git" # Public repo example
    example_repo = "https://github.com/pallets/flask.git" # Using Flask repo itself as a complex example (agent will likely fail without more specific instructions)
    example_base_branch = "main"

    # Example for a Node.js project
    # example_task = "In the Express app, add a new route GET /status that returns { status: 'ok' }. Add a basic Jest test for this route."
    # example_repo = "https://github.com/your-username/your-express-project.git"
    # example_base_branch = "master"


    print(f"Task: {example_task}")
    print(f"Repository: {example_repo}")
    print(f"Base Branch: {example_base_branch}")
    print("-" * 30)

    agent = CodeCraftAI()
    pull_request_url = agent.run(example_task, example_repo, example_base_branch)

    print("-" * 30)
    if pull_request_url:
        print(f"✅ Task Completed Successfully!")
        print(f"   Pull Request URL: {pull_request_url}")
    else:
        print(f"❌ Task Failed.")
        print(f"   Check the logs ({logging.getLogger().handlers[0].baseFilename if logging.getLogger().handlers else 'console'} and codecraft_log_*.log) for details.")

    print("-" * 30)
