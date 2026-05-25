from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# LIGHTWEIGHT MODEL
# ------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ------------------------------------------------
# CREATE EMBEDDINGS
# ------------------------------------------------

def create_embeddings(text_chunks):

    embeddings = model.encode(

        text_chunks,

        batch_size=4,

        show_progress_bar=False
    )

    return embeddings