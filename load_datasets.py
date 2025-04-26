import glob
import html
import os

from config import DATA_PATH #load path for data files in the file config

SEPARATOR = "\t"


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Argumentss:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip() #removes spaces, tabs, new lines etc FROM THE END of the text

    if '""' in text: #if we have useless double "  we clean them (""somthing"")
        if text[0] == text[-1] == '"':
            text = text[1:-1] #take the string without the first and last char (remove ")
        text = text.replace('\\""', '"')
        text = text.replace('""', '"') #double ""  replaced by "

    text = text.replace('\\""', '"')

    text = html.unescape(text) #convert HTML entities into characters ex. &lt; → <
    text = ' '.join(text.split()) #splits the text ignoring many spaces and ' '.join()
                                  # joins them with only one space
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines() #read all lines of the file (utf-8 for special chars)
    for _, line in enumerate(lines):
        columns = line.rstrip().split(SEPARATOR) #seperator = \t =  tab so we get each column value which is seperated by tab
        tweet_id = columns[0]
        sentiment = columns[1]
        text = columns[2:] #everything from the 3rd column and after (text)
        text = clean_text(" ".join(text)) #we set text as an element not a list of elements and clean it
                                          #in other words text becomes a string instead of a list 
        data[tweet_id] = (sentiment, text) #(emotion, text)
    return data


def load_from_dir(path):
    """
    Searches for all the .tsv and .txt files in a folder 

    """

    #search inside folders and subfolders
    files = glob.glob(path + "/**/*.tsv", recursive=True)
    files.extend(glob.glob(path + "/**/*.txt", recursive=True))

    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
               #dictionary will replace the duplicate key
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values()) #list of tuples (sentiment, text) --> dont care about the keys 




def load_Semeval2017A():
    """
    Loads data from dataset Semeval2017A
    """


    train = load_from_dir(os.path.join(DATA_PATH, "Semeval2017A/train_dev"))
    test = load_from_dir(os.path.join(DATA_PATH, "Semeval2017A/gold"))

    X_train = [x[1] for x in train]
    y_train = [x[0] for x in train]
    X_test = [x[1] for x in test]
    y_test = [x[0] for x in test]

    return X_train, y_train, X_test, y_test





def load_MR():
    with open(os.path.join(DATA_PATH, "MR/rt-polarity.pos"), encoding="utf-8", errors="replace") as f:
        pos = f.readlines()

    with open(os.path.join(DATA_PATH, "MR/rt-polarity.neg"), encoding="utf-8", errors="replace") as f:
        neg = f.readlines()

    pos = [x.strip() for x in pos]
    neg = [x.strip() for x in neg]

    pos_labels = ["positive"] * len(pos)
    neg_labels = ["negative"] * len(neg)

    split = 5000

    X_train = pos[:split] + neg[:split]
    y_train = pos_labels[:split] + neg_labels[:split]

    X_test = pos[split:] + neg[split:]
    y_test = pos_labels[split:] + neg_labels[split:]

    return X_train, y_train, X_test, y_test

