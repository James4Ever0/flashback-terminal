
from docarray import BaseDoc
from docarray.typing import NdArray, ID

class ToyDoc(BaseDoc):
  doc_id: ID
  text: str = ''
  embedding: NdArray[128]


from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

# Specify your workspace path
db = InMemoryExactNNVectorDB[ToyDoc](workspace='./workspace_path')

# Index a list of documents with random embeddings

doc_list = [ToyDoc(doc_id=str(i), text=f'toy doc {i}', embedding=np.random.rand(128)) for i in range(1000)]

db.index(inputs=DocList[ToyDoc](doc_list))

# try to index it twice?
db.index(inputs=DocList[ToyDoc](doc_list))

# Perform a search query
query = ToyDoc(text='query', embedding=np.random.rand(128))
results = db.search(inputs=DocList[ToyDoc]([query]), limit=10)

# Print out the matches
for m in results[0].matches:
  print(m)
