import hnswlib
import numpy as np

# for termux, use hnswlib, for desktop, use usearch (prebuilt)
# if using gcc for installing hnswlib:
# export CC=gcc
# export CXX=g++

dim = 128
num_elements = 10

data_elements = 5
# Generating sample data
data = np.float32(np.random.random((data_elements, dim)))
ids = np.arange(data_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16, allow_replace_deleted=True)

# Element insertion (can be called several times):
p.add_items(data, ids)
# let's replace old items?
# you can replace things. it will be overwritten.
p.add_items(np.float32(np.random.random((data_elements, dim))), ids, replace_deleted=True)

print("Ids:", ids)
print("Index size before mark deletion:", p.element_count) # 5
print("Index file size before mark deletion:", p.index_file_size())

p.mark_deleted(0)
p.mark_deleted(1)
p.mark_deleted(2)
p.mark_deleted(3)
p.mark_deleted(4)
# should we count that? how to get mark deleted element count via api?
# or you just log it with a separate database instead, after persisting index to disk

print("Index file size after mark deletion:", p.index_file_size())

print("Index size after mark deletion of 5 elems:", p.get_current_count()) # 5

p.add_items(np.float32(np.random.random((1, dim))), np.array([5]), replace_deleted=True) # will still increase index size?
p.add_items(np.float32(np.random.random((6, dim))), np.array([6,7,8,9,10,11]), replace_deleted=True) # will still increase index size?
# RuntimeError: The input label shape 1 does not match the input data vector shape 20 (ids and data shape mismatch.)
# RuntimeError: The number of elements exceeds the specified limit (data elements > index capacity)

print("Index file size after insertion:", p.index_file_size())


# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of the closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
p_copy = p
### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")

