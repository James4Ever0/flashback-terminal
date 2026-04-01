#!/usr/bin/env python3
"""
Async wrapper for HNSWIndex that makes all I/O bound operations non-blocking.
Uses asyncio loop.run_in_executor to offload blocking operations to thread pool.
"""

import asyncio
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import concurrent.futures
from sync import HNSWIndex


class AsyncHNSWIndex:
    """
    Async wrapper for HNSWIndex with non-blocking I/O operations.
    
    All methods that perform I/O (disk operations, heavy computations) are
    converted to async using loop.run_in_executor.
    
    Features:
    - Same API as HNSWIndex but with async/await
    - Non-blocking I/O operations
    - Thread pool for CPU-intensive operations
    - Preserves all original functionality
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async wrapper.
        
        Args:
            max_workers: Maximum number of worker threads in the thread pool
        """
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._index = None
        self._loop = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def close(self):
        """Close the thread pool executor"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def _get_loop(self):
        """Get or create event loop"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
        
    async def initialize(self, *args, **kwargs) -> 'AsyncHNSWIndex':
        """
        Initialize the underlying HNSWIndex.
        
        Args:
            *args: Positional arguments passed to HNSWIndex
            **kwargs: Keyword arguments passed to HNSWIndex
            
        Returns:
            Self for method chaining
        """
        loop = self._get_loop()
        self._index = await loop.run_in_executor(
            self._executor, 
            lambda: HNSWIndex(*args, **kwargs)
        )
        return self
        
    @property
    def dim(self) -> int:
        """Get vector dimension"""
        return self._index.dim if self._index else None
        
    @property
    def space(self) -> str:
        """Get distance space"""
        return self._index.space if self._index else None
        
    @property
    def max_elements(self) -> int:
        """Get maximum elements"""
        return self._index.max_elements if self._index else None
        
    @property
    def ef_construction(self) -> int:
        """Get ef_construction parameter"""
        return self._index.ef_construction if self._index else None
        
    @property
    def M(self) -> int:
        """Get M parameter"""
        return self._index.M if self._index else None
        
    @property
    def min_automatic_expansion_batch(self) -> int:
        """Get min_automatic_expansion_batch parameter"""
        return self._index.min_automatic_expansion_batch if self._index else None
        
    async def insert_batch(self, ids: List[int], vectors: List[np.ndarray]) -> None:
        """
        Async batch insertion of vectors.
        
        Args:
            ids: List of vector IDs
            vectors: List of vectors to insert
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        await loop.run_in_executor(
            self._executor,
            self._index.insert_batch,
            ids,
            vectors
        )
        
    async def insert(self, id: int, vector: np.ndarray) -> None:
        """
        Async single vector insertion.
        
        Args:
            id: Vector ID
            vector: Vector to insert
        """
        await self.insert_batch([id], [vector])
        
    async def delete_batch(self, ids: List[int]) -> int:
        """
        Async batch deletion of vectors.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of actually deleted items
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            self._index.delete_batch,
            ids
        )
        
    async def delete(self, id: int) -> int:
        """
        Async single vector deletion.
        
        Args:
            id: Vector ID to delete
            
        Returns:
            Number of deleted items (0 or 1)
        """
        return await self.delete_batch([id])
        
    async def get_batch(self, ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Async batch retrieval of vectors.
        
        Args:
            ids: List of vector IDs to retrieve
            
        Returns:
            Dictionary mapping IDs to vectors
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            self._index.get_batch,
            ids
        )
        
    async def get(self, id: int) -> np.ndarray:
        """
        Async single vector retrieval.
        
        Args:
            id: Vector ID to retrieve
            
        Returns:
            Vector data
        """
        result = await self.get_batch([id])
        if id not in result:
            raise KeyError(f"ID {id} not found")
        return result[id]
        
    async def search(
        self, 
        query_vector: Union[list[float], np.ndarray], 
        k: int, 
        ef: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Async k-NN search.
        
        Args:
            query_vector: Query vector (list or numpy array)
            k: Number of nearest neighbors to return
            ef: Optional ef parameter for search
            
        Returns:
            Tuple of (labels, distances)
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            self._index.search,
            query_vector,
            k,
            ef
        )
        
    async def get_all_ids(self) -> List[int]:
        """
        Async retrieval of all IDs in the index.
        
        Returns:
            List of all IDs
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            self._index.get_all_ids
        )
        
    async def resize(self, new_max_elements: int) -> None:
        """
        Async manual resize of the index.
        
        Args:
            new_max_elements: New maximum number of elements
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        await loop.run_in_executor(
            self._executor,
            self._index.resize,
            new_max_elements
        )
        
    async def save(self) -> None:
        """
        Async manual save of the index.
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        await loop.run_in_executor(
            self._executor,
            self._index._save
        )
        
    async def __len__(self) -> int:
        """
        Async get the number of vectors in the index.
        
        Returns:
            Number of vectors
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            len,
            self._index
        )
        
    async def __contains__(self, id: int) -> bool:
        """
        Async check if ID exists in the index.
        
        Args:
            id: ID to check
            
        Returns:
            True if ID exists, False otherwise
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        loop = self._get_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: id in self._index
        )
        
    async def batch_search(
        self, 
        query_vectors: List[Union[list[float], np.ndarray]], 
        k: int, 
        ef: Optional[int] = None
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Async batch search for multiple query vectors.
        
        Args:
            query_vectors: List of query vectors
            k: Number of nearest neighbors per query
            ef: Optional ef parameter for search
            
        Returns:
            List of (labels, distances) tuples for each query
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        # Run all searches concurrently
        tasks = [
            self.search(query_vector, k, ef) 
            for query_vector in query_vectors
        ]
        return await asyncio.gather(*tasks)
        
    async def batch_insert_with_progress(
        self, 
        ids: List[int], 
        vectors: List[np.ndarray], 
        batch_size: int = 1000,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Async batch insertion with progress reporting.
        
        Args:
            ids: List of vector IDs
            vectors: List of vectors to insert
            batch_size: Size of each insertion batch
            progress_callback: Optional callback function for progress updates
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
            
        total_items = len(ids)
        inserted = 0
        
        # Process in batches to allow progress reporting
        for i in range(0, total_items, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            
            await self.insert_batch(batch_ids, batch_vectors)
            inserted += len(batch_ids)
            
            if progress_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    progress_callback,
                    inserted,
                    total_items
                )


# Convenience function for quick async index creation
async def create_async_hnsw_index(
    db_path: str,
    dim: int,
    space: str = "l2",
    max_elements: int = 10000,
    ef_construction: int = 200,
    M: int = 16,
    min_automatic_expansion_batch: int = 1000,
    max_workers: int = 4
) -> AsyncHNSWIndex:
    """
    Convenience function to create and initialize an AsyncHNSWIndex.
    
    Args:
        db_path: Database path
        dim: Vector dimension
        space: Distance space
        max_elements: Initial maximum elements
        ef_construction: HNSW ef_construction parameter
        M: HNSW M parameter
        min_automatic_expansion_batch: Minimum expansion batch size
        max_workers: Thread pool workers
        
    Returns:
        Initialized AsyncHNSWIndex
    """
    async_index = AsyncHNSWIndex(max_workers=max_workers)
    await async_index.initialize(
        db_path=db_path,
        dim=dim,
        space=space,
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=M,
        min_automatic_expansion_batch=min_automatic_expansion_batch
    )
    return async_index


# Example usage and testing
async def example_usage():
    """Example of how to use AsyncHNSWIndex"""
    
    # Create and initialize index
    async with AsyncHNSWIndex() as index:
        await index.initialize(
            db_path="test_async_db",
            dim=64,
            space="l2",
            max_elements=10
        )

        # print(index._index)
        # breakpoint()
        
        # Insert data
        num_items = 100
        ids = list(range(num_items))
        vectors = [np.random.randn(64).astype(np.float32) for _ in ids]
        
        print("Inserting vectors...")
        await index.insert_batch(ids, vectors)
        print(f"Inserted {len(await index.get_all_ids())} vectors")
        
        # Search
        query_vector = np.random.randn(64).astype(np.float32)
        labels, distances = await index.search(query_vector, k=5)
        print(f"Search results: {labels}")
        
        # Batch search
        query_vectors = [np.random.randn(64).astype(np.float32) for _ in range(10)]
        batch_results = await index.batch_search(query_vectors, k=3)
        print(f"Batch search completed: {len(batch_results)} queries")
        
        # Delete some items
        deleted_count = await index.delete_batch(ids[:10])
        print(f"Deleted {deleted_count} items")
        
        # Get vector
        try:
            vector = await index.get(ids[50])
            print(f"Retrieved vector shape: {vector.shape}")
        except KeyError:
            print("Vector not found")
            
    print("Example completed!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
