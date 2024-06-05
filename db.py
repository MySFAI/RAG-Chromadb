from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from main import ml_papers_dict, get_completion
import chromadb
from tqdm.auto import tqdm
import random
from chromadb.config import Settings

settings = Settings(allow_reset=True)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
mistral_llm = "mistral-7b-instruct-4k"

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = embedding_model.encode(input)
        return batch_embeddings.tolist()

embed_fn = MyEmbeddingFunction()

# Initialize the chromadb directory, and client.
client = chromadb.PersistentClient(path="./chromadb", settings=chromadb.config.Settings(allow_reset=True))
client. reset() 

# create collection
collection = client.get_or_create_collection(
    name=f"ml-papers-nov-2023"
)

# Generate embeddings, and index titles in batches
batch_size = 50

# loop through batches and generated + store embeddings
for i in tqdm(range(0, len(ml_papers_dict), batch_size)):

    i_end = min(i + batch_size, len(ml_papers_dict))
    batch = ml_papers_dict[i : i + batch_size]

    # Replace title with "No Title" if empty string
    batch_titles = [str(paper["Title"]) if str(paper["Title"]) != "" else "No Title" for paper in batch]
    batch_ids = [str(sum(ord(c) + random.randint(1, 10000) for c in paper["Title"])) for paper in batch]
    batch_metadata = [dict(url=paper["PaperURL"],
                           abstract=paper['Abstract'])
                           for paper in batch]

    # generate embeddings
    batch_embeddings = embedding_model.encode(batch_titles)

    # upsert to chromadb
    collection.upsert(
        ids=batch_ids,
        metadatas=batch_metadata,
        documents=batch_titles,
        embeddings=batch_embeddings.tolist(),
    )

collection = client.get_or_create_collection(
    name=f"ml-papers-nov-2023",
    embedding_function=embed_fn
)

# user query
user_query = "S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for Large Language Models"

# query for user query
results = collection.query(
    query_texts=[user_query],
    n_results=10,
)


# concatenate titles into a single string
short_titles = '\n'.join(results['documents'][0])
print(short_titles)

prompt_template = f'''[INST]

Your main task is to generate 5 SUGGESTED_TITLES based for the PAPER_TITLE

You should mimic a similar style and length as SHORT_TITLES but PLEASE DO NOT include titles from SHORT_TITLES in the SUGGESTED_TITLES, only generate versions of the PAPER_TILE.

PAPER_TITLE: {user_query}

SHORT_TITLES: {short_titles}

SUGGESTED_TITLES:

[/INST]
'''

responses = get_completion(prompt_template, model=mistral_llm, max_tokens=2000)
suggested_titles = ''.join([str(r) for r in responses])

# Print the suggestions.
print("Model Suggestions:")
print(suggested_titles)
