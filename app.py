import streamlit as st

st.set_page_config(
    page_title="Movie Quote Recommender",
    layout="centered"
)

import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: 
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.avif")



import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("./movie_quotes.csv")

bert_emb  = np.load("./bert_embeddings.npy")
sbert_emb = np.load("./sbert_embeddings.npy")
mpnet_emb = np.load("./mpnet_embeddings.npy")


@st.cache_resource
def load_models():
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    mpnet_model = SentenceTransformer("all-mpnet-base-v2", device=device)

    return bert_tokenizer, bert_model, sbert_model, mpnet_model

bert_tokenizer, bert_model, sbert_model, mpnet_model = load_models()


def encode_bert(texts):
    with torch.no_grad():
        encoded = bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = bert_model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.cpu().numpy()

def encode_sbert(texts):
    return sbert_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def encode_mpnet(texts):
    return mpnet_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def recommend_ensemble(input_quote, top_n=5, per_model_k=10):
    q_bert  = encode_bert([input_quote])
    q_sbert = encode_sbert([input_quote])
    q_mpnet = encode_mpnet([input_quote])

    sims = {
        "bert":  cosine_similarity(q_bert,  bert_emb).flatten(),
        "sbert": cosine_similarity(q_sbert, sbert_emb).flatten(),
        "mpnet": cosine_similarity(q_mpnet, mpnet_emb).flatten(),
    }

    vote_count = {}
    score_sum = {}

    for model_sims in sims.values():
        top_idx = np.argsort(model_sims)[::-1][:per_model_k]
        for i in top_idx:
            vote_count[i] = vote_count.get(i, 0) + 1
            score_sum[i] = score_sum.get(i, 0) + model_sims[i]

    ranked = sorted(
        vote_count.keys(),
        key=lambda i: (vote_count[i], score_sum[i] / vote_count[i]),
        reverse=True
    )[:top_n]

    results = df.iloc[ranked].copy()
    results["votes"] = [vote_count[i] for i in ranked]
    results["avg_score"] = [score_sum[i] / vote_count[i] for i in ranked]
    results["year"] = results["year"].astype(str)

    results.drop(
        columns=["type", "clean_quote", "clean_quote_light", "clean_quote_strict"],
        inplace=True,
        errors="ignore"
    )

    return results.reset_index(drop=True)


st.title("🎬 Movie Quote Recommender")
st.write(
    "Masukkan potongan dialog film, lalu sistem akan mencari kutipan "
    "yang paling mirip secara makna menggunakan ensemble BERT, SBERT, dan MPNet."
)

user_input = st.text_input("Enter a movie quote:")

if user_input:
    with st.spinner("Finding similar quotes..."):
        recommendations = recommend_ensemble(user_input, top_n=5)

    top_result = recommendations.iloc[0]

    st.subheader("⭐ Best Match")
    st.markdown(
        f"""
        **{top_result['movie']} ({top_result['year']})**  
        > *“{top_result['quote']}”*

        🔍 Similarity score rata-rata: **{top_result['avg_score']:.3f}**  
        🗳️ Dipilih oleh **{top_result['votes']} dari 3 model**
        """
    )

    st.subheader("Top Recommendations")

    for _, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"### 🎞️ {row['movie']} ({row['year']})")
            st.markdown(f"> *“{row['quote']}”*")
            st.markdown(
                f"""
                **Votes:** {row['votes']} / 3  
                **Average Similarity Score:** {row['avg_score']:.3f}
                """
            )
            st.divider()

