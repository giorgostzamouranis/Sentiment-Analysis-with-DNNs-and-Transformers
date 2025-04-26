from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

DATASET = 'MR'
#PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
#PRETRAINED_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
PRETRAINED_MODEL = 'textattack/bert-base-uncased-SST-2'

"""
DATASET = 'Semeval2017A'
#PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
#PRETRAINED_MODEL = 'finiteautomata/bertweet-base-sentiment-analysis'
PRETRAINED_MODEL = 'yiyanghkust/finbert-tone'
"""


LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    
    'distilbert-base-uncased-finetuned-sst-2-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative'
    },



    'textattack/bert-base-uncased-SST-2':{
        'LABEL_1': 'positive',
        'LABEL_0': 'negative'
    },


    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },

    'finiteautomata/bertweet-base-sentiment-analysis': {
        'POS': 'positive',
        'NEG': 'negative',
        'NEU': 'neutral',
    },

    'yiyanghkust/finbert-tone':{
        'Neutral': 'neutral',
        'Positive': 'positive',
        'Negative': 'negative',

    }



}

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
    le.fit(list(set(y_train))) #learn the labels 
    y_train = le.transform(y_train) #convert them to numbers
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

    """
    sentiment_pipeline("I love this!")
        |
        |

    [{'label': 'POSITIVE', 'score': 0.998}]

    it returns a list with a dictionary with predicted label and confidence score
    
    """


    y_pred = []
    for x in tqdm(X_test): #tqdm for bar (visualization)
        # TODO: Main-lab-Q6 - get the label using the defined pipeline 
        pred = sentiment_pipeline(x)
        label = pred[0]['label']
        y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

    y_pred = le.transform(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
