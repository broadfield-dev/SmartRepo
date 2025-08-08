import os
import pandas as pd
from pathlib import Path
import datetime
import subprocess
import platform
import sys

def get_directory_contents(path_str: str) -> (pd.DataFrame):
    try:
        if sys.platform == "win32" and len(path_str) == 2 and path_str[1] == ':':
            path_str += "\\"
        path = Path(path_str)
        if not path.is_dir():
            return pd.DataFrame(), f"Error: '{path_str}' is not a valid directory."
        items = []
        for item in path.iterdir():
            try:
                stat = item.stat()
                is_dir = item.is_dir()
                items.append({
                    "Name": item.name, "Type": "📁 Folder" if is_dir else "📄 File",
                    "Size (bytes)": stat.st_size if not is_dir else "","Modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    "Permissions": oct(stat.st_mode)[-3:], "Full Path": str(item.resolve())})
            except (FileNotFoundError, PermissionError):
                items.append({ "Name": item.name, "Type": "❓ Inaccessible", "Size (bytes)": "N/A", "Modified": "N/A", "Permissions": "N/A", "Full Path": str(item)})
            except Exception:
                 items.append({ "Name": item.name, "Type": "❓ Unknown", "Size (bytes)": "N/A", "Modified": "N/A", "Permissions": "N/A", "Full Path": str(item)})
        df = pd.DataFrame(items)
        if not df.empty:
            df = df.sort_values(by=['Type', 'Name'], ascending=[False, True])
        return df, f"Currently viewing: {path.resolve()}"
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"


def get_file_content(filepath: str) -> (str):
    try:
        path = Path(filepath)
        if path.is_dir(): return "# This is a directory. Please select a file to view its content.", ""
        try:
            with open(path, 'rb') as f:
                if b'\x00' in f.read(1024):
                    return f"# File '{path.name}' appears to be a binary file.\n# Cannot display content.", filepath
        except Exception:
             return f"# Could not read file '{path.name}' to determine type.", filepath
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, filepath
    except Exception as e:
        return f"# Error reading file: {str(e)}", filepath

def get_pip_freeze() -> str:
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"Error running 'pip freeze': {str(e)}"

def get_disk_usage() -> str:
    system = platform.system()
    try:
        if system == "Windows": command = ['wmic', 'logicaldisk', 'get', 'caption,size,freespace']
        else: command = ['df', '-h']
        proc = subprocess.run(command, capture_output=True, text=True, check=True, shell=(system=="Windows"))
        return proc.stdout
    except FileNotFoundError:
        return f"Error: Command not found. Your system ('{system}') may not have the required tool."
