
# IT Support Ticket NLP Pipeline

> **First NLP Project | Sprints Scholarship - Milestone 3**

A complete end-to-end Natural Language Processing project tackling IT support ticket automation across three progressive phases: classification, sequence modeling, and question answering.

---

## 📌 Project Overview

This project simulates a real-world IT support automation pipeline. The goal is to process raw customer support tickets and extract meaningful insights—starting from simple categorization and advancing to contextual question answering.

| Phase | Task | Best Model | Accuracy |
|-------|------|------------|----------|
| 1 | Ticket Classification | TF-IDF + Logistic Regression | **85%** |
| 2 | Contextual Classification | LSTM | **83%** |
| 3 | Question Answering | BERT (Extractive QA) | N/A |

---

## 🗂️ Repository Structure

```
├── data_pipeline.ipynb          # Phase 1: Preprocessing, TF-IDF, Word2Vec
├── lstm_classifier.py           # Phase 2: RNN & LSTM models
├── Transformer_QA.ipynb         # Phase 3: BERT Question Answering
├── evaluation_report_m1.txt     # Phase 1 results & analysis
├── evaluation_report_m2.txt     # Phase 2 results & RNN/LSTM comparison
├── architecture_justification.md # Transformer architecture deep dive
├── tfidf_vectorizer.pkl         # Saved TF-IDF model
└── README.md                    # You are here
```

---

## 🚀 Phase 1: Baseline Classification & Feature Engineering

### What I Learned
- Building a complete text preprocessing pipeline from scratch
- The difference between **TF-IDF** (keyword frequency) and **Word2Vec** (semantic embeddings)
- How class imbalance affects model performance

### Preprocessing Steps (NLTK)
1. Lowercasing
2. Punctuation removal
3. Tokenization
4. Stopword removal
5. Lemmatization (reducing words to root form)

### Models & Results

| Model | Accuracy | Best Class (F1) | Worst Class (F1) |
|-------|----------|-----------------|------------------|
| TF-IDF + Logistic Regression | **85%** | Purchase (92%) | Administrative rights (77%) |
| Word2Vec + Logistic Regression | 57% | Hardware (62%) | Internal Project (36%) |

### Key Insight
> For short, domain-specific IT tickets, keyword frequency (TF-IDF) significantly outperforms averaged word vectors (Word2Vec), which lose critical distinguishing terms.

### Identified Risks
- Class imbalance biases models toward majority classes
- Word2Vec requires better tuning and data balancing

---

## 🔁 Phase 2: Sequence Modeling with RNN & LSTM

### The Problem with RNNs
Standard RNNs suffer from the **vanishing gradient problem**—during backpropagation, gradients shrink exponentially as they travel through many time steps. The model effectively "forgets" words from the beginning of long tickets.

### How LSTM Solves This

LSTM uses three intelligent gates to manage memory:

| Gate | Function |
|------|----------|
| **Forget Gate** | Decides what information to discard from previous states |
| **Input Gate** | Decides what new information to store |
| **Output Gate** | Decides what part of the memory to output |

### Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Simple RNN | 62% | N/A |
| LSTM | **83%** | 84% |

### Per-Class Performance (LSTM)

| Best Class | F1-Score | Worst Class | F1-Score |
|------------|----------|-------------|----------|
| Storage | 89% | Administrative rights | 76% |

### LSTM vs. Phase 1 Best Model

| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression (Phase 1) | 85% |
| LSTM (Phase 2) | 83% |

> The 2% difference suggests keyword-based features are very strong for this dataset. LSTM's advantage would be more pronounced on longer or more complex texts where word order and context matter more.

---

## 🧠 Phase 3: Transformer-Based Question Answering

### Moving Beyond Classification

Instead of just classifying tickets, this phase answers specific questions like:
> *"What is the primary technical component or system mentioned?"*

### Architecture Choice: BERT (Encoder-Only)

| Architecture | Directionality | Best For |
|--------------|----------------|----------|
| **BERT** | Bidirectional (left-to-right AND right-to-left) | Understanding, Question Answering |
| **GPT** | Unidirectional (left-to-right only) | Text Generation |

### Why BERT Over LSTM?

| Feature | LSTM | BERT (Transformer) |
|---------|------|---------------------|
| Processing | Sequential (one word at a time) | Parallel (all words at once) |
| Long-range context | Fades due to vanishing gradient | Maintained via Self-Attention |
| Bidirectional | No (only forward) | Yes (both directions) |

### The Self-Attention Mechanism

Self-attention allows every word to "attend" to every other word in the sequence. When processing a question like *"What technical component is failing?"*, BERT can directly link the word "component" to relevant terms like "Kubernetes", "RAID controller", or "firmware" anywhere in the ticket.

### Positional Encoding

Since Transformers process all words in parallel, they have no natural sense of order. **Positional encoding** adds information about each token's position in the sequence. BERT uses **learned absolute positional embeddings** (trainable, limit of 512 tokens).

### Model Used
```
deepset/bert-large-uncased-whole-word-masking-finetuned-squad
```
from Hugging Face Transformers library.

### Sample Results

| Ticket Context | Extracted Answer |
|----------------|------------------|
| "Outage with centralized account management portal" | `centralized account management portal` |
| "Issue with Laser Printer after macOS update" | `printer firmware` |
| "Dashboard crash traced to MySQL 8.0 database" | `data overflow or an analytics software error` |
| "Smart Home Center problem after firmware upgrade" | `firmware upgrade` |
| "Critical outage in Kubernetes orchestration" | `barcode scanners, raid controllers` |

---

## 📊 Comparison Across All Models

| Phase | Model | Accuracy | Key Strength |
|-------|-------|----------|--------------|
| 1 | TF-IDF + Logistic Regression | **85%** | Fast, interpretable, strong on keywords |
| 1 | Word2Vec + Logistic Regression | 57% | Semantic understanding (needs tuning) |
| 2 | Simple RNN | 62% | Handles sequences poorly (vanishing gradient) |
| 2 | LSTM | 83% | Remembers long-range context |
| 3 | BERT (QA) | N/A | Deep understanding, answer extraction |

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|----------|-------------------|
| Preprocessing | NLTK (punkt, stopwords, wordnet) |
| Feature Extraction | Scikit-learn (TF-IDF), Gensim (Word2Vec) |
| Classification | Scikit-learn (Logistic Regression) |
| Sequence Models | TensorFlow / Keras (RNN, LSTM) |
| Transformers | Hugging Face (BERT, Tokenizers) |
| Data Handling | Pandas, NumPy |
| Evaluation | Classification report, Accuracy, F1, Precision, Recall |

---

## 📈 Key Takeaways

1. **Start simple** — TF-IDF with Logistic Regression is a powerful baseline
2. **Word2Vec requires careful tuning** — averaged vectors lose information
3. **LSTMs solve the vanishing gradient problem** but aren't always better than a good baseline
4. **Transformers are revolutionary** — self-attention changes everything for long-range dependencies
5. **Choose architecture based on your goal**:
   - Classification → TF-IDF or LSTM
   - Answer extraction → BERT (encoder-only)
   - Text generation → GPT (decoder-only)

---

## ⚠️ Known Issues & Future Improvements

| Issue | Proposed Solution |
|-------|-------------------|
| Class imbalance | Apply SMOTE or class weights |
| Word2Vec underperforming | Increase vector size, tune window, more epochs |
| LSTM marginal improvement | Hyperparameter tuning, bidirectional LSTM |
| BERT not fine-tuned | Fine-tune on domain-specific IT ticket data |

---

## 🏁 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/it-support-nlp.git
cd it-support-nlp
```

### 2. Install dependencies
```bash
pip install nltk pandas numpy scikit-learn gensim tensorflow transformers torch
```

### 3. Run Phase 1 (Classification)
Open `data_pipeline.ipynb` in Jupyter Notebook and run all cells.

### 4. Run Phase 2 (LSTM)
```bash
python lstm_classifier.py
```

### 5. Run Phase 3 (BERT QA)
Open `Transformer_QA.ipynb` and run all cells.

---

## 🙏 Acknowledgments

- **Sprints Scholarship** — for providing structured learning and this milestone project
- **Hugging Face** — for pretrained models and the Transformers library
- **Kaggle** — for the IT Support Ticket dataset

---

## 📬 Connect With Me

[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📝 License

This project is for educational purposes as part of the Sprints scholarship program.

---

*Made with ❤️ as my first NLP project*
