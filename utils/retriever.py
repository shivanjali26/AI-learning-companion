import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    k=2
):

    question_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(question_embedding),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks