import numpy as np
import faiss

features = np.load("features.npy").astype("float32")

dim = features.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(features)

faiss.write_index(index, "index.faiss")
print("✅ FAISS index built successfully!")
