import numpy as np
import threading
import yaml
from typing import List, Optional, Tuple, Dict, Union
import atexit
import traceback
from pathlib import Path
try:
    from usearch.index import Index
except ImportError:
    raise ImportError("usearch package is required. Install with: pip install usearch")
 
class USearchIndex:
    """
    Thread-safe USearch index wrapper with persistence.
 
    Features:
    - Persistent vector store using USearch engine
    - Delete by id (removes from index)
    - Batch insert/delete operations
    - Saves index and metadata after every write operation
    - On load, checks dimension and metric against saved metadata; raises if mismatch
    - Thread lock for concurrent access
    - Support for various metrics (l2sq, ip, cos, haversine, etc.)
    - Configurable data types (f32, f16, i8, etc.)
    """
    @staticmethod
    def _resolve_np_dtype(usearch_dtype:str):
        if usearch_dtype == "f32":
            return np.float32
        elif usearch_dtype == "f16":
            return np.float16
        elif usearch_dtype == "i8":
            return np.int8
        else:
            raise ValueError(f"Unsupported dtype: {usearch_dtype}")
    def __init__(
        self,
        db_path: Union[str, Path],
        dim: int,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
        multi: bool = False,
    ):
        """
        Initialize or load the database.
 
        Args:
            db_path: Base path for files (e.g., "db" → db.usearch, db.yaml)
            dim: Vector dimension (checked against saved file if loading)
            metric: Distance metric ('l2sq', 'ip', 'cos', 'haversine', etc.)
            dtype: Data type for vectors ('f32', 'f16', 'i8', etc.)
            connectivity: Number of neighbors per graph node
            expansion_add: Control recall of indexing
            expansion_search: Control quality of search
            multi: Allow multiple vectors per key
        """
        db_path = Path(db_path)
        if not db_path.is_dir():
            db_path.mkdir(parents=True, exist_ok=True)
 
        self.db_path = db_path
        self.dim = dim
        self.metric = metric
        self.dtype = dtype
        np_dtype = self._resolve_np_dtype(dtype)
        self.np_dtype = np_dtype
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self.multi = multi
        self.rw_lock = threading.Lock()
 
        self.index: Optional[Index] = None
 
        # Try to load existing database
        index_file = db_path / "index.usearch"
        yaml_file = db_path / "metadata.yaml"
 
        self._need_save = False
 
        atexit.register(self._atexit_save)
 
        if index_file.exists() and yaml_file.exists():
            # Load metadata and check dimensions/metric
            with open(yaml_file, 'r') as f:
                meta: dict = yaml.safe_load(f)
            saved_dim = meta.get('dim')
            saved_metric = meta.get('metric')
            saved_dtype = meta.get('dtype')
 
            if saved_dim != dim:
                raise ValueError(f"Dimension mismatch: file has {saved_dim}, given {dim}")
            if saved_metric != metric:
                raise ValueError(f"Metric mismatch: file has {saved_metric}, given {metric}")
            if saved_dtype != dtype:
                raise ValueError(f"Dtype mismatch: file has {saved_dtype}, given {dtype}")
 
            # Load the index
            self.index = Index.restore(str(index_file))
 
            # Verify loaded index matches our parameters
            if self.index.ndim != dim:
                raise ValueError(f"Loaded index dimension {self.index.ndim} doesn't match expected {dim}")
 
        else:
            self._need_save = True
            # Create new index
            self.index = Index(
                ndim=dim,
                metric=metric,
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                multi=multi,
            )
            self._save()  # Save initial state
 
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
 
    def _atexit_save(self) -> None:
        with self.rw_lock:
            self._save()
 
    def _save(self) -> None:
        """Save the index and metadata to disk. Assumes write lock is held."""
        if not self._need_save: 
            return
 
        index_file = self.db_path / "index.usearch"
        yaml_file = self.db_path / "metadata.yaml"
 
        # Save index
        try:
            if hasattr(self, 'index') and self.index is not None:
                self.index.save(str(index_file))
            else:
                print("Index not initialized")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to save index: {e}")
 
        try:
            # Save metadata
            metadata = {
                'dim': self.dim,
                'metric': self.metric,
                'dtype': self.dtype,
                'connectivity': self.connectivity,
                'expansion_add': self.expansion_add,
                'expansion_search': self.expansion_search,
                'multi': self.multi,
            }
            with open(yaml_file, 'w') as f:
                yaml.dump(metadata, f)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to save metadata: {e}")
 
        self._need_save = False
 
    # -------------------------------------------------------------------------
    # Public CRUD operations (with locking)
    # -------------------------------------------------------------------------
 
    def insert(self, id: int, vector: np.ndarray) -> None:
        """Insert a single vector. Write operation."""
        return self.insert_batch([id], [vector])
 
    def insert_batch(
        self, ids: List[int], vectors: List[np.ndarray]
    ) -> None:
        """Insert a batch of vectors. Write operation."""
        with self.rw_lock:
            self._need_save = True
            if len(ids) != len(vectors):
                raise ValueError("ids and vectors must have same length")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate ids in batch")
 
            # Convert vectors to numpy array
            vec_array = np.array(vectors, dtype=self.np_dtype)
            if vec_array.shape[1] != self.dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vec_array.shape[1]}")
 
            # Insert vectors
            self.index.add(ids, vec_array)
            self._save()
 
    def delete_batch(self, ids: List[int]) -> int:
        """Delete multiple vectors by ids. Returns actual deleted count. Write operation."""
        with self.rw_lock:
            self._need_save = True
            actual_deleted = 0
 
            for id in ids:
                try:
                    # USearch supports removal by key
                    self.index.remove(id)
                    actual_deleted += 1
                except Exception:
                    # ID might not exist, continue
                    pass
 
            if actual_deleted > 0:
                self._save()
            return actual_deleted
 
    def delete(self, id: int) -> None:
        """Delete a vector by id. Write operation."""
        return self.delete_batch([id])
 
    def get_batch(self, ids: List[int]) -> Dict[int, np.ndarray]:
        """Retrieve multiple vectors by ids. Returns dict of id->vector. Read operation."""
        with self.rw_lock:
            result = {}
            for id in ids:
                try:
                    vector = self.index[id]
                    result[id] = vector
                except Exception:
                    # ID not found, skip
                    pass
            return result
 
    def get(self, id: int) -> np.ndarray:
        """Retrieve the vector associated with the given id. Read operation."""
        with self.rw_lock:
            try:
                return self.index[id]
            except Exception:
                raise KeyError(f"ID {id} not found")
 
    def search(
        self, query_vector: Union[list[float], np.ndarray], k: int
    ) -> Tuple[List[int], List[float]]:
        """
        Search for the k nearest neighbors. Read operation.
 
        Returns:
            Tuple of (keys, distances)
        """
        with self.rw_lock:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=self.np_dtype)
 
            if query_vector.shape != (self.dim,):
                raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {query_vector.shape[0]}")
 
            # Search for nearest neighbors
            matches = self.index.search(query_vector, k)
 
            # Extract keys and distances
            keys = [match.key for match in matches]
            distances = [match.distance for match in matches]
 
            return keys, distances
 
    def get_all_ids(self) -> List[int]:
        """Return list of all ids. Read operation."""
        with self.rw_lock:
            # USearch doesn't have a direct get_all_ids method
            # We'll need to track this ourselves or reconstruct from metadata
            # For now, return empty list - this would need to be implemented
            # based on specific usearch capabilities or external tracking
            return []
 
    def __bool__(self) -> bool:
        return True
 
    def __len__(self) -> int:
        """Number of vectors. Read operation."""
        with self.rw_lock:
            # USearch doesn't expose direct count, this would need tracking
            # Return 0 for now - implement based on your tracking needs
            return 0
 
    def __contains__(self, id: int) -> bool:
        """Check if id exists. Read operation."""
        with self.rw_lock:
            try:
                _ = self.index[id]
                return True
            except Exception:
                return False
 
 
# Example usage
if __name__ == "__main__":
    # Create new database
    db = USearchIndex(db_path="test_usearch_db", dim=128, metric="cos", dtype="f32")
 
    # Insert a batch
    ids = [1, 2, 3, 4]
    vecs = [np.random.randn(128).astype(np.float32) for _ in ids]
    db.insert_batch(ids, vecs)
 
    # Search
    query = np.random.randn(128).astype(np.float32)
    nearest_keys, distances = db.search(query, k=2)
    print("Nearest keys:", nearest_keys)
    print("Distances:", distances)
 
    # Delete
    db.delete(3)
 
    # Check containment
    print("Contains 1:", 1 in db)
    print("Contains 3:", 3 in db)
 
    # Get vector
    try:
        vector = db.get(1)
        print("Retrieved vector shape:", vector.shape)
    except KeyError as e:
        print(e)
 
    # Close and reopen (simulate loading)
    del db
    db2 = USearchIndex(db_path="test_usearch_db", dim=128, metric="cos", dtype="f32")
    print("After reload, contains 1:", 1 in db2)
    print("After reload, contains 3:", 3 in db2)