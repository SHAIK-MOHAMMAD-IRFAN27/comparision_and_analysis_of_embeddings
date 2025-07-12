# comparision_and_analysis_of_embeddings
#  Word Representations Comparison: BoW, TF-IDF, CBOW, and Skip-Gram

##  Project Overview

This project explores and compares **four different word representation techniques** using a small custom text corpus:

1. **Bag of Words (BoW)**
2. **TF-IDF (Term Frequency–Inverse Document Frequency)**
3. **CBOW (Continuous Bag of Words)**
4. **Skip-Gram**

Our goal is to understand how these representations differ in:
- Structure (sparse vs. dense)
- Interpretability
- Ability to capture semantics
- Performance in visual clustering

---

##  Word Representation Techniques Explained

### 1.  Bag of Words (BoW)

- Counts the frequency of each word in the document.
- Simple, fast, but ignores word order and semantics.

> Example:
> - Doc1: "The cat sat"
> - BoW vector: `[the:1, cat:1, sat:1]`

---

### 2.  TF-IDF (Term Frequency–Inverse Document Frequency)

- Weighs words by frequency in a document but penalizes common words across documents.
- Reduces importance of common words like "the", "is".

> Example:
> - "the" has low IDF; "neural" or "quantum" may have high IDF if rare.

In the notebook:
- `TfidfVectorizer` from `sklearn` is used
- A TF-IDF matrix is generated and top-weighted words per doc are shown

---

### 3.  CBOW (Continuous Bag of Words)

- Predicts the **center word** from its surrounding **context**.
- Learns dense, low-dimensional embeddings from co-occurrence patterns.

> Example:  
> For window = 2, and text: “The cat sat on the mat”,  
> CBOW learns to predict `sat` from `[the, cat, on, the]`.

---

### 4.  Skip-Gram

- Does the reverse of CBOW: given the center word, predicts the surrounding context.
- Especially good for learning representations of rare words.

> Example:  
> From the word `sat`, predict `[the, cat, on, the]`.

---

##  Dataset and Preprocessing

- A IMDB movie reviews dataset used
- Tokenized using regex-based word extraction.
- A vocabulary is built with `<PAD>` and `<UNK>` tokens.
- For CBOW and Skip-Gram:
  - Word pairs are generated using sliding windows
  - Batches are prepared for model training

---

##  Model Architecture (CBOW & Skip-Gram)

Implemented from scratch using PyTorch.

Both use:
- `nn.Embedding` to map word indices to vectors
- A linear projection layer with softmax
- `CrossEntropyLoss` as the objective

---

##  Training Details

- Optimizer: `torch.optim.SGD`
- Epochs: 5
- Training loss per epoch is logged
- Two separate models are trained: one for CBOW, one for Skip-Gram

---

##  Visualization

- **PCA** and **t-SNE** are used to project embeddings into 2D space
- Selected word embeddings (e.g., `['good', 'bad', 'movie', ...]`) are plotted
- Cosine similarity matrices are shown to measure semantic similarity

---

##  Example Observations

###  BoW & TF-IDF:
- Easy to understand
- BoW treats all words equally (no down-weighting)
- TF-IDF down-weights frequent terms like "the", "is"
- Vectors are high-dimensional and sparse

###  CBOW & Skip-Gram:
- Dense, low-dimensional (e.g., 50 or 100D)
- Capture context semantics
- In visual plots:
  - Skip-Gram showed better word clusters
  - Words like `love`, `great`, `funny` were closer to `good`

---

##  Sample Code Snippets

### Cosine Similarity of Embeddings
```python
def get_topk_similar(word, embeddings, k=5):
    idx = word2idx[word]
    vec = embeddings[idx]
    similarities = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), embeddings)
    topk = torch.topk(similarities, k+1).indices[1:].tolist()
    return [idx2word[i] for i in topk]
