import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #not GPU


import numpy as np
import evaluate
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A
import torch



#PRETRAINED_MODEL = 'bert-base-cased'
DATASET = 'MR' 
PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
#PRETRAINED_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
#PRETRAINED_MODEL = 'textattack/bert-base-uncased-SST-2'


"""
DATASET = 'Semeval2017A'
#PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
#PRETRAINED_MODEL = 'finiteautomata/bertweet-base-sentiment-analysis'
PRETRAINED_MODEL = 'yiyanghkust/finbert-tone'
"""





metric = evaluate.load("accuracy") #instead of sklearn.accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) #takes class with highest score
    return metric.compute(predictions=predictions, references=labels) #returns {'accuracy': 0.87}
    #return {"accuracy": accuracy_score(labels, predictions)}


def tokenize_function(examples): # examples = {"text": [sentence1, sentence2,...]}
    return tokenizer(examples["text"], padding="max_length", truncation=True) #truncation -> if the text is big,we cut it in the max length that the model allows


def prepare_dataset(X, y): #convert raw data (lists X,y) to huggingface Dataset
    texts, labels = [], []
    for text, label in zip(X, y):
        texts.append(text)
        labels.append(label)

    return Dataset.from_dict({'text': texts, 'label': labels})

#hugging face takes this type of input



if __name__ == '__main__':



    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # prepare datasets
    train_set = prepare_dataset(X_train, y_train) #convert to the type huggingface takes them
    test_set = prepare_dataset(X_test, y_test)

    # define model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    """
    tokenizer("I love NLP!")
    # -> {'input_ids': [...], 'attention_mask': [...]}

    """


    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=n_classes)
    
    model.to("cpu")

    #this downloads the model from hugging face and sets a classification as last layer


    # tokenize datasets
    tokenized_train_set = train_set.map(tokenize_function)
    tokenized_test_set = test_set.map(tokenize_function)

    """
    # ΠΡΙΝ την κλήση του map (raw dataset):
    # train_set[0] =
    # {
    #     'text': "I love this movie!",
    #     'label': 1
    # }

    # META την κλήση του map με τη tokenize_function:
    # tokenized_train_set[0] =
    # {
    #     'input_ids': [101, 1045, 2293, 2023, 3185, 999, 102, 0, 0, ...],
    #     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
    #     'label': 1
    # }

    #  Τα input_ids είναι οι αριθμητικοί δείκτες των tokens (λέξεων)
    #  Το attention_mask δείχνει ποιοι είναι "πραγματικοί" tokens (1) και ποιοι είναι padding (0)
    #  Η 'label' παραμένει ίδια, απλώς πλέον τα κείμενα είναι σε μορφή που "καταλαβαίνει" το transformer

    """



    # TODO: Main-lab-Q7 - remove this section once you are ready to execute on a GPU
    #  create a smaller subset of the dataset
    n_samples = 40
    small_train_dataset = tokenized_train_set.shuffle(
        seed=42).select(range(n_samples))
    small_eval_dataset = tokenized_test_set.shuffle(
        seed=42).select(range(n_samples))

    # TODO: Main-lab-Q7 - customize hyperparameters once you are ready to execute on a GPU
    # training setup
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=8,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # train
    trained_model = trainer.train()
  
    #final testing to compare the accuracy with the not fine tuned model
    # Predict on full test set (X_test, y_test)
    predictions = trainer.predict(tokenized_test_set)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    
    acc = accuracy_score(y_test, pred_labels)
    print(f"\nFinal test accuracy: {acc:.4f}")


