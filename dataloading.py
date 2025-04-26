from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer #okenizer της NLTK, φτιαγμένος ειδικά για tweets 
import numpy as np



class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """


    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        tokenizer = TweetTokenizer()
        self.data = [list(map(lambda x:x.lower(), tokenizer.tokenize(x))) for x in X]       
        """"
        tokenizer.tokenize(x) → χωρίζει το tweet σε tokens (π.χ. ["I", "love", "#pizza"])
        
        map(lambda x: x.lower(), ...) → κάνει κάθε token μικρό (lowercase) ωστε GOOD =Good
        
        all these, for each tweet so list from lists
        """

        self.tokenized_data = self.data
        self.labels = y        
        self.word2idx = word2idx #every token is cnoverted to numbers {"i": 1, "love": 2, "pizza": 3, "<unk>": 0} that points to the index of the word in the embeddings
        self.max_length = 35 #if a tweet is smaller, then zeros (padding)
        #length of the tokens of a text (words)

        # EX2
        #raise NotImplementedError




    def __getitem__(self, index):
        
        example = self.tokenized_data[index] #example = ["i", "love", "pizza", "!"]
        # map tokens to ids according to word2idx
        example = [self.word2idx.get(token, self.word2idx['<unk>']) for token in example]
        label = self.labels[index]
        length = len(example) 

        # zero padding using the maximum length from initialization
        # or truncation in case a larger example is found
        if length < self.max_length:
            example += [0] * (self.max_length - length) 
        else:
            example = example[:self.max_length] # if the tweet is bigger, we lose info but
                                                # in emotion detection the first words have the most info
            
        return np.array(example), label, length #[ [token_ids], "label", length ]


    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)
