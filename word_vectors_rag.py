"""
Word vectors + simple RAG with MiniLM (sentence-transformers).

Install:  pip install sentence-transformers
Run:      python word_vectors_rag.py
"""
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim vectors


# ---------------------------------------------------------------
# Part 1: show the vector of every word in a sentence
# ---------------------------------------------------------------
def word_vectors(sentence):
    # One contextual vector per token, including [CLS] and [SEP]
    token_vecs = model.encode(sentence, output_value="token_embeddings")

    # MiniLM splits some words into sub-word pieces ("embeddings" -> "em", "##bed", "##ding", "##s").
    # word_ids() tells us which word each token belongs to, so we can average the pieces back into words.
    enc = model.tokenizer(sentence)
    words = {}
    for token_idx, word_idx in enumerate(enc.word_ids()):
        if word_idx is None:  # skip [CLS] / [SEP]
            continue
        start, end = enc.word_to_chars(word_idx)
        words.setdefault(word_idx, (sentence[start:end], []))[1].append(token_vecs[token_idx])

    return [(word, sum(vecs) / len(vecs)) for word, vecs in words.values()]


sentence = "RAG retrieves relevant documents before generating an answer"
print(f"Sentence: {sentence}\n")
for word, vec in word_vectors(sentence):
    print(f"{word:<12} dim={len(vec)}  first 5 values: {vec[:5].numpy().round(4)}")


