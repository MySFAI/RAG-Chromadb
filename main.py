import fireworks.client
import os
import dotenv
import chromadb
import json
from tqdm.auto import tqdm
import pandas as pd
import random

# you can set envs using Colab secrets
dotenv.load_dotenv()

fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")


# load dataset from data/ folder to pandas dataframe
# dataset contains column names

ml_papers = pd.read_csv("ml-potw-10232023.csv", header=0)

# remove rows with empty titles or descriptions
ml_papers = ml_papers.dropna(subset=["Title", "Description"])

ml_papers.head()
ml_papers_dict = ml_papers.to_dict(orient="records")

def get_completion(prompt, model=None, max_tokens=50):

	fw_model_dir = "accounts/fireworks/models/"

	if model is None:
		model = fw_model_dir + "llama-v2-7b"
	else:
		model = fw_model_dir + model

	completion = fireworks.client.Completion.create(
		model=model,
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=0
	)

	return completion.choices[0].text

if __name__ == "__main__":
	print(ml_papers_dict[0])
