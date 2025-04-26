# Sentiment Analysis with DNNs and Transformers

This project explores sentiment analysis using Deep Neural Networks (Baseline DNN, LSTM, Attention, Transformer Encoder) and transfer learning with pre-trained Transformer models from Huggingface.  
It was developed as part of the **Speech and Natural Language Processing** course (Spring 2024-25, NTUA).

**Authors:**
- Tzamouranis Georgios (03121141)
- Katsaidonis Nikolaos (03121868)  


---

## Project Structure

```
sentiment-analysis-lab2/
├── attention.py               # Custom attention and transformer encoder modules
├── dataloading.py             # Custom PyTorch Dataset (tokenization, padding)
├── finetune_pretrained.py     # Fine-tune Huggingface Transformer models
├── load_datasets.py           # Load MR and Semeval datasets
├── main.py                    # Main script to train various architectures
├── models.py                  # Definitions for DNN, LSTM, Attention, Transformers
├── training.py                # Utilities for training and evaluation loops
├── transfer_pretrained.py     # Transfer learning using Huggingface pipelines
├── utils/                     # Helper scripts (data loading, embeddings loading, etc.)
├── saved_models/              # Directory to save trained model checkpoints
├── embeddings/                # Directory for pre-trained embeddings (GloVe, FastText)
└── requirements.txt           # Python dependencies
```

---

## Implemented Architectures

### Baseline DNN

- Mean-Max pooling sentence representation.
- Fully connected classification layers.

### LSTM / Bi-Directional LSTM

- Captures sequential context.
- Early stopping implementation to avoid overfitting.

### Simple Self-Attention

- Improved training speed compared to LSTM.
- Captures important words through self-attention.

### Multi-Head Attention

- Multiple attention heads focusing on different sentence aspects simultaneously.

### Transformer Encoder

- Multiple stacked attention and feedforward layers.
- Hyperparameter tuning (heads and layers).

### Pre-trained Transformers

- Leveraging Huggingface models:
  - `siebert/sentiment-roberta-large-english`
  - `textattack/bert-base-uncased-SST-2`
  - `cardiffnlp/twitter-roberta-base-sentiment`

### Fine-Tuning Pretrained Transformers

- Further training Huggingface models specifically on MR and Semeval datasets.

---

## Results Summary

| Dataset  | Model                          | Accuracy  |
|----------|--------------------------------|-----------|
| MR       | Baseline DNN (mean pooling)    | ~70.8%    |
| MR       | LSTM (bidirectional)           | ~74.5%    |
| MR       | Self-Attention                 | ~71.4%    |
| MR       | Multi-Head Attention           | ~72.2%    |
| MR       | Transformer Encoder            | ~73.7%    |
| MR       | Pretrained (Siebert)           | **92.6%** |
| Semeval  | Transformer Encoder (best)     | ~62.8%    |
| Semeval  | Pretrained (Cardiffnlp)        | **72.4%** |

> Fine-tuning results varied significantly based on available data and computational resources.

---

## Key Takeaways

- Pre-trained embeddings like **GloVe** significantly enhance performance.
- Transformer models generally outperform simpler neural architectures.
- Prompt design significantly affects accuracy, especially when using models like ChatGPT.
- Early stopping and dropout techniques effectively control overfitting.

---

## Acknowledgements

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Huggingface Transformers](https://huggingface.co/models)
- NTUA Speech and Natural Language Processing course (Spring 2024-25)

---

> This project was developed for educational purposes and is not intended for production use.

---

