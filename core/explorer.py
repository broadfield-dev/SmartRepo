import os
import datetime
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import urllib.parse
from . import repo_loader

class SemanticExplorer:
    def __init__(self, db_path="./chroma_db", collection_name="filesystem_index"):
        print("Initializing SemanticExplorer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("SemanticExplorer initialized.")
        self.is_cancelled = False

    def get_status(self) -> str:
        count = self.collection.count()
        if count > 0:
            return f"Persistent index loaded with {count} items."
        return "Index is empty. Build the index to get started."

    def _get_file_snippet(self, path, max_len=500) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # Basic check for binary files
                if '\x00' in f.read(1024): return ""
                f.seek(0)
                return f.read(max_len)
        except Exception:
            return ""

    def index_directory(self, path_or_url: str, progress_callback=None):
        """
        Builds or updates the index from a local directory path or a remote repository URL.
        This is a generator that yields status updates.
        """
        self.is_cancelled = False
        if "github.com" in path_or_url or "huggingface.co" in path_or_url:
            yield from self._index_repository(path_or_url, progress_callback)
        else:
            yield from self._index_local_directory(path_or_url, progress_callback)

    def _index_repository(self, repo_url: str, progress_callback=None):
        """Indexes a remote repository from a URL."""
        if progress_callback: progress_callback(0, desc="Fetching repository file list...")
        yield "Fetching repository file list..."

        files, content_base_url, error = repo_loader.get_repo_files(repo_url)
        if error:
            yield f"Error: {error}"
            return

        total_files = len(files)
        if progress_callback: progress_callback(0.05, desc=f"Found {total_files} files to process.")
        yield f"Found {total_files} files to process."

        batch_size = 50
        for i in range(0, total_files, batch_size):
            if self.is_cancelled: break
            
            progress_fraction = (i / total_files) * 0.95 + 0.05 # Scale progress from 5% to 100%
            status_message = f"Processing batch {i//batch_size + 1}... ({min(i+batch_size, total_files)}/{total_files})"
            if progress_callback: progress_callback(progress_fraction, desc=status_message)
            yield status_message

            batch_files = files[i:i+batch_size]
            docs, metadatas, ids = [], [], []

            for file_meta in batch_files:
                file_path = file_meta['path']
                # Correctly join URL parts, ensuring file_path is properly encoded
                file_url = urllib.parse.urljoin(content_base_url, urllib.parse.quote(file_path))
                
                content, err = repo_loader.get_remote_file_content(file_url)
                if err:
                    print(f"Skipping {file_path}: {err}")
                    continue
                
                snippet = content[:500]
                doc = f"Type: File. Path: {file_path.replace('/', ' ')}. Tree: {' > '.join(Path(file_path).parts)}. Content Snippet: {snippet}"
                unique_id = f"repo::{repo_url}::{file_path}"
                
                docs.append(doc)
                metadatas.append({
                    "full_path": file_url, "relative_path": file_path, "is_dir": False,
                    "size_bytes": file_meta.get('size', len(content)),
                    "modified_time": datetime.datetime.now().timestamp(), # No reliable modified time from API
                    "source_type": "repository", "repo_url": repo_url
                })
                ids.append(unique_id)

            if docs:
                self.collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

        final_count = self.collection.count()
        if self.is_cancelled:
            yield f"Build cancelled. The database now contains {final_count} items."
        else:
            if progress_callback: progress_callback(1, desc="Complete!")
            yield f"Index build complete. The database now contains {final_count} items."

    def _index_local_directory(self, root_path: str, progress_callback=None):
        if not os.path.isdir(root_path):
            yield "Error: Provided path is not a valid directory."
            return
        if progress_callback:
            progress_callback(0, desc="Scanning directories...")
        yield "Scanning directories..."
        all_paths = []
        for root, dirs, files in os.walk(root_path):
            if self.is_cancelled: break
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.vscode', 'node_modules', '.idea', 'chroma_db']]
            for name in files + dirs:
                all_paths.append(os.path.join(root, name))
        if self.is_cancelled:
            yield f"Build cancelled. DB size: {self.collection.count()}"
            return
        total_paths = len(all_paths)
        if progress_callback:
            progress_callback(0.05, desc=f"Scan complete. Found {total_paths} items.")
        yield f"Scan complete. Found {total_paths} items to process."
        batch_size = 128
        for i in range(0, total_paths, batch_size):
            if self.is_cancelled: break
            progress_fraction = i / total_paths
            status_message = f"Processing batch {i//batch_size + 1}... ({min(i+batch_size, total_paths)}/{total_paths})"
            if progress_callback:
                progress_callback(progress_fraction, desc=status_message)
            yield status_message
            batch_paths = all_paths[i:i+batch_size]
            docs, metadatas, ids = [], [], []
            for path_str in batch_paths:
                try:
                    stat = os.stat(path_str)
                    is_dir = os.path.isdir(path_str)
                    relative_path = os.path.relpath(path_str, root_path)
                    doc = f"Type: {'Folder' if is_dir else 'File'}. Path: {relative_path.replace(os.sep, ' ')}. Tree: {' > '.join(Path(relative_path).parts)}. "
                    if not is_dir: doc += f"Content Snippet: {self._get_file_snippet(path_str)}"
                    docs.append(doc)
                    metadatas.append({
                        "full_path": path_str, "relative_path": relative_path, "is_dir": is_dir,
                        "size_bytes": stat.st_size, "modified_time": stat.st_mtime,
                        "source_type": "local"
                    })
                    unique_id = f"local::{path_str}"
                    ids.append(unique_id)
                except FileNotFoundError:
                    continue
            if docs:
                self.collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
        final_count = self.collection.count()
        if self.is_cancelled:
            yield f"Build cancelled. The database now contains {final_count} items."
        else:
            if progress_callback:
                progress_callback(1, desc="Complete!")
            yield f"Index build complete. The database now contains {final_count} items."

    def search(self, query: str, n_results: int = 20, metadata_filters: dict = None) -> list[dict]:
        if self.collection.count() == 0: return []
        
        db_filters = metadata_filters.copy() if metadata_filters else {}
        path_contains_filter = None
        
        def extract_path_filter(conditions):
            nonlocal path_contains_filter
            remaining_conditions = []
            for cond in conditions:
                if 'relative_path' in cond and '$contains' in cond['relative_path']:
                    path_contains_filter = cond['relative_path']['$contains']
                else:
                    remaining_conditions.append(cond)
            return remaining_conditions

        if '$and' in db_filters:
            db_filters['$and'] = extract_path_filter(db_filters['$and'])
            if not db_filters['$and']:
                del db_filters['$and']
        elif 'relative_path' in db_filters and '$contains' in db_filters['relative_path']:
            path_contains_filter = db_filters['relative_path']['$contains']
            del db_filters['relative_path']

        query_params = {
            "query_embeddings": self.model.encode([query]).tolist(),
            "n_results": min(n_results * 5, self.collection.count())
        }
        if db_filters:
            query_params["where"] = db_filters
        
        results = self.collection.query(**query_params)
        
        output = []
        if not results['ids'][0]: return []

        for i, dist in enumerate(results['distances'][0]):
            meta = results['metadatas'][0][i]
            
            if path_contains_filter and path_contains_filter not in meta['relative_path']:
                continue

            output.append({
                "similarity": 1 - dist,
                "path": meta['relative_path'],
                "full_path": meta['full_path'],
                "type": "📁 Folder" if meta.get('is_dir', False) else "📄 File",
                "size": meta['size_bytes'] if not meta.get('is_dir', False) else None,
                "modified": datetime.datetime.fromtimestamp(meta['modified_time'])
            })
            
            if len(output) >= n_results:
                break
                
        return output

    def clear_index(self) -> int:
        count = self.collection.count()
        if count > 0:
            ids_to_delete = self.collection.get(include=[])['ids']
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
        return count
    
    def cancel_indexing(self):
        self.is_cancelled = True
