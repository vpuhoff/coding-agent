import os
import logging
import argparse
from codecraft.orchestrator.orchestrator import Orchestrator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

parser = argparse.ArgumentParser(description="CodeCraft AI – Autonomous Coding Assistant")
parser.add_argument("repo_url", help="URL of the Git repository to work on")
parser.add_argument("base_branch", help="Name of the base branch to create the new branch from")
parser.add_argument("task", help="Textual description of the development task to perform")
args = parser.parse_args()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY is not set.")
    exit(1)
if not os.getenv("GITHUB_TOKEN"):
    print("Error: GITHUB_TOKEN is not set.")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

orchestrator = Orchestrator()
try:
    pr_url = orchestrator.run(args.task, args.repo_url, args.base_branch)
    print(f"Pull request created: {pr_url}")
except Exception as e:
    logging.error(f"Agent execution failed: {e}")
    exit(1)
