import hnswlib
import numpy as np
import threading
import yaml
from typing import List, Optional, Tuple, Dict, Union
import atexit
import traceback
from pathlib import Path

class HNSWIndex:
    """
    Thread-safe HNSW index wrapper with persistence.

    Features:
    - Initial max_elements = 10,000 (or custom), expands by auto_expand_step when needed.
    - Delete by id (marks as deleted in index).
    - Always replaces deleted entries at insertion time.
    - Batch insert with automatic expansion.
    - Saves index and metadata after every write operation.
    - On load, checks dimension and space against the saved metadata; raises if mismatch.
    - Thread lock for concurrent access.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        dim: int,
        space: str = "l2",
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
        min_automatic_expansion_batch: int = 1000,
    ):
        """
        Initialize or load the database.

        Args:
            db_path: Base path for files (e.g., "db" → db.hnsw, db.yaml, db.vectors.pkl)
            dim: Vector dimension (checked against saved file if loading)
            space: Distance metric ('l2', 'ip', 'cosine') (checked against saved file)
            max_elements: Initial maximum elements (used only when creating new database)
            ef_construction: HNSW construction parameter
            M: HNSW M parameter
            allow_replace_deleted: If True, deleted slots can be reused when adding new items
            min_automatic_expansion_batch: Minimum max element limit expansion batch when automatic index expansion happens.
        """
        db_path = Path(db_path)
        if not db_path.is_dir():
            db_path.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.dim = dim
        self.space = space
        self.ef_construction = ef_construction
        self.M = M
        self.allow_replace_deleted = True  # Always True
        self.rw_lock = threading.Lock()
        self.min_automatic_expansion_batch = min_automatic_expansion_batch

        self.index: Optional[hnswlib.Index] = None
        self.max_elements = max_elements

        # Try to load existing database
        index_file = db_path / "index.hnsw"
        yaml_file = db_path / "metadata.yaml"

        self._need_save = False

        atexit.register(self._atexit_save) # save the db for whatever reason at exit.

        if index_file.exists() and yaml_file.exists():
            # Load metadata and check dimensions/space
            with open(yaml_file, 'r') as f:
                meta:dict = yaml.safe_load(f)
            saved_dim = meta.get('dim')
            saved_space = meta.get('space')
            if saved_dim != dim:
                raise ValueError(f"Dimension mismatch: file has {saved_dim}, given {dim}")
            if saved_space != space:
                raise ValueError(f"Space mismatch: file has {saved_space}, given {space}")

            # Load the index
            self.index = hnswlib.Index(space=space, dim=dim)
            self.index.load_index(str(index_file), allow_replace_deleted=True)

            # Update internal max_elements
            self.max_elements = self.index.get_max_elements()

        else:
            self._need_save = True
            # Create new index
            self.index = hnswlib.Index(space=space, dim=dim)
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
                allow_replace_deleted=True,
            )
            self.max_elements = max_elements
            self._save()  # Save initial state

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _atexit_save(self) -> None:
        with self.rw_lock():
            self._save()
            
    def _save(self) -> None:
        """Save the index and metadata to disk. Assumes write lock is held."""
        if not self._need_save: return

        index_file = self.db_path / "index.hnsw"
        yaml_file = self.db_path / "metadata.yaml"

        # Save index
        try:
            if hasattr(self, 'index'):
                self.index.save_index(str(index_file))
            else:
                print("Index not initilaized")
        except:
            traceback.print_exc()
            print("Failed to save index")

        try:
            # Save metadata (including deleted indices)
            metadata = {
                'dim': self.dim,
                'space': self.space,
                'ef_construction': self.ef_construction,
                'M': self.M,
                'allow_replace_deleted': self.allow_replace_deleted,
                'min_automatic_expansion_batch': self.min_automatic_expansion_batch,
            }
            with open(yaml_file, 'w') as f:
                yaml.dump(metadata, f)
        except:
            traceback.print_exc()
            print("Failed to save metadata")
        
        self._need_save =False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _resize(self, new_max_elements: int) -> None:
        """Resize the index. Assumes write lock is held."""
        self.index.resize_index(new_max_elements)
        self.max_elements = new_max_elements

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
            self._need_save=True
            if len(ids) != len(vectors):
                raise ValueError("ids and vectors must have same length")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate ids in batch")

            # Convert vectors to numpy array
            vec_array = np.array(vectors, dtype=np.float32)
            if vec_array.shape[1] != self.dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vec_array.shape[1]}")

            # Try to insert, expand aggressively if RuntimeError occurs
            try:
                self.index.add_items(vec_array, ids, replace_deleted=True)
            except RuntimeError as e:
                if "exceeds the specified limit" in str(e) or "capacity" in str(e).lower():
                    # Aggressively expand by the number of items we're trying to insert
                    expansion_size = len(ids)
                    expansion_size = max(expansion_size, self.min_automatic_expansion_batch)
                    new_max_elements = self.max_elements + expansion_size
                    self._resize(new_max_elements)
                    
                    # Try insertion again
                    try:
                        self.index.add_items(vec_array, ids, replace_deleted=True)
                    except RuntimeError as e2:
                        raise RuntimeError(f"Failed to insert {len(ids)} items even after expanding index to {new_max_elements}: {e2}")
                else:
                    raise RuntimeError(f"Insertion failed with unexpected error: {e}")

            self._save()

    def delete_batch(self, ids: List[int]) -> int:
        """Delete multiple vectors by ids. Returns actual deleted count. Write operation."""
        with self.rw_lock:
            self._need_save = True
            actual_deleted = 0
            for id in ids:
                try:
                    self.index.mark_deleted(id)
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
            # Get vectors from index
            vectors = self.index.get_items(ids)
            return {id: vectors[i] for i, id in enumerate(ids)}

    def get(self, id: int) -> np.ndarray:
        """Retrieve the vector associated with the given id. Read operation."""
        with self.rw_lock:
            try:
                vectors = self.index.get_items([id])
                return vectors[0]
            except Exception as e:
                if "Label not found" in str(e):
                    raise KeyError(f"ID {id} not found")
                else:
                    raise e
    def search(
        self, query_vector: Union[list[float], np.ndarray], k: int, ef: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Search for the k nearest neighbors. Read operation.
        Optionally set ef for this query.
        """
        with self.rw_lock:
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
            if query_vector.shape != (self.dim,):
                raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {query_vector.shape[0]}")

            # Store original ef to restore later
            original_ef = self.index.ef

            try:
                if ef is not None:
                    self.index.set_ef(ef)
                labels, distances = self.index.knn_query(query_vector.reshape(1, -1), k=k)
                return labels[0].tolist(), distances[0].tolist()
            finally:
                # Restore original ef
                if ef is not None:
                    self.index.set_ef(original_ef)

    def get_all_ids(self) -> List[int]:
        """Return list of all ids. Read operation."""
        with self.rw_lock:
            return list(self.index.get_ids_list())

    def resize(self, new_max_elements: int) -> None:
        """Manually resize the index. Write operation."""
        with self.rw_lock:
            self._need_save = True
            self._resize(new_max_elements)
            self._save()
    
    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        """Number of vectors. Read operation."""
        with self.rw_lock:
            return len(self.index.get_ids_list())

    def __contains__(self, id: int) -> bool:
        """Check if id exists. Read operation."""
        with self.rw_lock:
            return id in self.index.get_ids_list()


# Example usage
if __name__ == "__main__":
    # Create new database
    db = HNSWIndex(db_path="test_db", dim=128, space="l2", max_elements=2)

    # Insert a batch
    ids = [1, 2, 3, 4]
    vecs = [np.random.randn(128).astype(np.float32) for _ in ids]
    db.insert_batch(ids, vecs)

    # Search
    query = np.random.randn(128).astype(np.float32)
    nearest, dist = db.search(query, k=2)
    print("Nearest ids:", nearest)

    # Delete
    db.delete(3)

    # Get all ids
    print("All ids:", db.get_all_ids())

    # Close and reopen (simulate loading)
    del db
    db2 = HNSWIndex(db_path="test_db", dim=128, space="l2")  # Should load existing
    print("After reload, all ids:", db2.get_all_ids())
    print("After reload, max item size:", db2.max_elements)