import ast
import json

import numpy as np
import pandas as pd
import requests
import tiktoken
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

url = "https://api.openai.com/v1/embeddings"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


def read_json_as_pd(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def most_similar(embeddings, query_embedding, top_n=5):
    """
    Find the most similar embeddings to the query_embedding in the given embeddings.
    """
    similarities = cosine_similarity(query_embedding, embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]

    return sorted_indices[:top_n]


def save_hn_embeddings(hn_data):
    hn_df = read_json_as_pd(hn_data)
    hn_df = hn_df.dropna()
    hn_df["combined"] = (
        "Title: "
        + hn_df.header.str.strip()
        + "; Content: "
        + hn_df.description.str.strip()
    )

    encoding = tiktoken.get_encoding(embedding_encoding)

    hn_df["n_tokens"] = hn_df.combined.apply(lambda x: len(encoding.encode(x)))
    hn_df = hn_df[hn_df.n_tokens <= max_tokens]

    hn_df["embedding"] = hn_df.combined.apply(
        lambda x: get_embedding(x, engine=embedding_model)
    )
    hn_df.to_csv("src/data/hn_embeddings.csv", index=False)


def save_portfolio_embeddings(portfolio_data):
    portfolio_df = read_json_as_pd(portfolio_data)
    portfolio_df = portfolio_df.dropna()
    portfolio_df["combined"] = (
        "Name: "
        + portfolio_df.name.str.strip()
        + "; Sector: "
        + portfolio_df.sector.str.strip()
        + " "
        + portfolio_df.sector_hover_card.str.strip()
        + " "
        + portfolio_df.sector_company_page.str.strip()
        + "; Introduction: "
        + portfolio_df.introduction_company_page.str.strip()
        + "; Description: "
        + portfolio_df.description_company_page.str.strip()
        + "; Location: "
        + portfolio_df.country.str.strip()
    )
    encoding = tiktoken.get_encoding(embedding_encoding)
    portfolio_df["n_tokens"] = portfolio_df.combined.apply(
        lambda x: len(encoding.encode(x))
    )
    portfolio_df = portfolio_df[portfolio_df.n_tokens <= max_tokens]
    portfolio_df["embedding"] = portfolio_df.combined.apply(
        lambda x: get_embedding(x, engine=embedding_model)
    )
    portfolio_df.to_csv("src/data/portfolio_embeddings.csv", index=False)


def read_hn_embeddings(csv_data):
    hn_df = pd.read_csv(csv_data)
    hn_df["embedding"] = (
        hn_df["embedding"]
        .apply(ast.literal_eval)
        .apply(lambda x: [float(i) for i in x])
    )
    return hn_df


def read_portfolio_embeddings(csv_data):
    portfolio_df = pd.read_csv(csv_data)
    portfolio_df["embedding"] = (
        portfolio_df["embedding"]
        .apply(ast.literal_eval)
        .apply(lambda x: [float(i) for i in x])
    )
    return portfolio_df


def find_top_matches(df_a, df_b, top_n=10):
    # Calculate cosine similarity matrix
    cos_sim_matrix = cosine_similarity(
        np.stack(df_a["embedding"]), np.stack(df_b["embedding"])
    )

    # Find the highest similarity value for each row in df_a
    highest_similarity_values = np.max(cos_sim_matrix, axis=1)

    # Get the indices of the top_n rows in df_a with the highest similarity values
    top_indices = np.argsort(highest_similarity_values)[-top_n:]

    print("\nTop Matches:\n")
    print("-------------------------------------------------------------")
    for index in top_indices[::-1]:  # [::-1] to start from the highest similarity
        matching_b_index = np.argmax(
            cos_sim_matrix[index]
        )  # The index in df_b for the top match
        company_name = df_a.iloc[index]["header"].split("|")[0].strip()
        print(f"Company from HN: {company_name}")
        print(f"Top Match from the portfolio:")
        print(f"    Name   : {df_b.iloc[matching_b_index]['name']}")
        print(f"    Country: {df_b.iloc[matching_b_index]['country']}")
        print(f"    Sector : {df_b.iloc[matching_b_index]['sector']}")
        print(f"Similarity Value: {highest_similarity_values[index]:.4f}")
        print("-------------------------------------------------------------")


def find_top_matches_agg(df_a, df_b, top_n=15):
    # Calculate cosine similarity matrix
    cos_sim_matrix = cosine_similarity(
        np.stack(df_a["embedding"]), np.stack(df_b["embedding"])
    )

    # Get the sum of similarities for each entity in df_a to all entities in df_b
    sim_sum = np.sum(cos_sim_matrix, axis=1)

    # Get the indices of the top_n most similar entities from df_a
    top_indices = np.argsort(sim_sum)[-top_n:]

    print("Top Matches Aggregate:\n")
    print("================================================================")
    # For each of the top matches, find which rows in df_b were most influential in their ranking
    for index in top_indices[::-1]:  # [::-1] is to start from the highest similarity
        matching_b_indices = np.argsort(cos_sim_matrix[index])[
            -3:
        ]  # Top 3 from df_b for each top entity in df_a
        company_name = df_a.iloc[index]["header"].split("|")[0].strip()
        print(f"\nPotential investment match: {company_name}")
        print("Similar to the following companies in the current portfolio:")
        for b_index in matching_b_indices[::-1]:
            print(f"    - Company: {df_b.iloc[b_index]['name']}")
            print(f"      Country: {df_b.iloc[b_index]['country']}")
            print(f"      Sector : {df_b.iloc[b_index]['sector']}\n")
        print("----------------------------------------------------------------")
    print("\n")


# main function
if __name__ == "__main__":
    save_hn_embeddings("src/data/hn-dump.json")
    save_portfolio_embeddings("src/data/portfolio-dump.json")
    # hn_df = read_hn_embeddings("hn_embeddings.csv")
    # portfolio_df = read_portfolio_embeddings("portfolio_embeddings.csv")

    # find_top_matches(hn_df, portfolio_df, top_n=15)
