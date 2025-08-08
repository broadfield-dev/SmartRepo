# core/repo_loader.py
import requests
import os
from huggingface_hub import HfApi
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import urllib.parse

# Setup logger
logger = logging.getLogger(__name__)

# Constants
GITHUB_API = "https://api.github.com/repos/"
EXCLUDE_EXTENSIONS = {'.lock', '.log', '.env', '.so', '.o', '.a', '.dll', '.exe', '.ipynb', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.otf'}
EXCLUDE_FILENAMES = {'.gitignore', '.DS_Store', '.gitattributes'}
EXCLUDE_PATTERNS = {'__pycache__/', '.git/', 'node_modules/', 'dist/', 'build/', '.vscode/', '.idea/'}

# GitHub API Headers
HEADERS = {"Accept": "application/json"}
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'

def is_excluded(filepath: str) -> bool:
    path = Path(filepath)
    if path.name.lower() in EXCLUDE_FILENAMES or path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    normalized_path = path.as_posix() + '/'
    if any(pattern in normalized_path for pattern in EXCLUDE_PATTERNS):
        return True
    return False

def _get_github_files(owner: str, repo: str) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str]]:
    """Fetches file list from GitHub, returns (files, default_branch, error_str)."""
    try:
        repo_info_url = f"{GITHUB_API}{owner}/{repo}"
        repo_response = requests.get(repo_info_url, headers=HEADERS, timeout=10)
        repo_response.raise_for_status()
        default_branch = repo_response.json()['default_branch']

        branch_info_url = f"{GITHUB_API}{owner}/{repo}/branches/{default_branch}"
        branch_response = requests.get(branch_info_url, headers=HEADERS, timeout=10)
        branch_response.raise_for_status()
        tree_sha = branch_response.json()['commit']['commit']['tree']['sha']

        tree_url = f"{GITHUB_API}{owner}/{repo}/git/trees/{tree_sha}?recursive=1"
        tree_response = requests.get(tree_url, headers=HEADERS, timeout=30)
        tree_response.raise_for_status()

        files = [
            {"path": item['path'], "size": item.get('size', 0)}
            for item in tree_response.json()['tree']
            if item['type'] == 'blob'
        ]
        return files, default_branch, None
    except requests.RequestException as e:
        error_message = f"Error fetching from GitHub API: {e}. "
        if 'rate limit' in str(e).lower() and not GITHUB_TOKEN:
            error_message += "You may have hit the rate limit. Please set a GITHUB_TOKEN environment variable."
        return None, None, error_message

def _get_hf_files(owner: str, repo: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Fetches file list from Hugging Face, returns (files, error_str)."""
    try:
        api = HfApi(token=os.getenv('HF_TOKEN'))
        repo_info = api.repo_info(repo_id=f'{owner}/{repo}', repo_type="space")
        files = [{"path": f.rfilename, "size": f.size} for f in repo_info.siblings if f.blob_id is not None]
        return files, None
    except Exception as e:
        return None, f"Error fetching from Hugging Face Hub: {e}"

def get_repo_files(url: str) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str]]:
    """
    Gets a list of file metadata from a repository URL.
    Returns (list_of_files, base_url_for_content, error_str).
    list_of_files is a list of dicts with 'path' and 'size'.
    """
    try:
        parts = url.strip().rstrip('/').split('/')
        if len(parts) < 2:
            return None, None, "Invalid repository URL format."
        owner, repo = parts[-2], parts[-1]
        is_hf = "huggingface.co" in url.lower()

        if is_hf:
            files, error = _get_hf_files(owner, repo)
            base_url = f"https://huggingface.co/spaces/{owner}/{repo}/raw/main/"
        else: # Assuming GitHub
            files, default_branch, error = _get_github_files(owner, repo)
            base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/"

        if error:
            return None, None, error
        
        if not files:
            error_msg = f"No files found in {'Hugging Face Space' if is_hf else 'GitHub repository'}."
            if not is_hf and not GITHUB_TOKEN:
                 error_msg += " If this is a private repo or you've hit a rate limit, please set a GITHUB_TOKEN."
            return None, None, error_msg

        filtered_files = [f for f in files if not is_excluded(f['path'])]
        return filtered_files, base_url, None
    except Exception as e:
        logger.error(f"Error processing repo URL '{url}': {e}", exc_info=True)
        return None, None, f"An unexpected error occurred: {str(e)}"

def get_remote_file_content(file_url: str, max_size=1_000_000) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches content of a remote file. Returns (content, error_str).
    Skips files that are too large.
    """
    try:
        with requests.get(file_url, stream=True, timeout=15) as r:
            r.raise_for_status()
            
            content_length = r.headers.get('content-length')
            if content_length and int(content_length) > max_size:
                return None, f"File is too large ({int(content_length)/1024:.1f} KB > {max_size/1024:.1f} KB)."

            content = b''
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_size:
                    return None, f"File is too large (exceeded {max_size/1024:.1f} KB while streaming)."

            if b'\x00' in content[:1024]:
                return None, "File appears to be binary."
            
            return content.decode('utf-8', errors='ignore'), None
    except requests.RequestException as e:
        return None, f"Network error fetching file: {e}"
    except Exception as e:
        return None, f"Error reading remote file: {e}"
