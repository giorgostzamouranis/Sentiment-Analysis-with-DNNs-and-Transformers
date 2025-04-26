################################################################

# RUNNING WITH KERNEL Python 3.8.20 (slp3) ---> anaconda 

                                                                    
################################################################









import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from models import LSTM
from attention import SimpleSelfAttentionModel,MultiHeadAttentionModel,TransformerEncoderModel
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from training import torch_train_val_split
from early_stopper import EarlyStopper


########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False #with false, model do not change the embeddings
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM) #idx2word → αντίστροφο λεξικό
 # το word2idx αντιστοιχιζει ενα index στη καθε λεξη και αυτο το index αντιστοιχει στο index της λεξης στα embedings


# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# Save the original (string) labels for the EX1 prints
y_train_orig = y_train.copy()

# convert data labels from strings to integers
label_encoder = LabelEncoder()
label_encoder.fit(y_train) #convert labels to numbers [0,1,2,...]
y_train = label_encoder.transform(y_train)  # EX1 #we could do fit_transform
y_test = label_encoder.transform(y_test)  # EX1
n_classes = label_encoder.classes_.size  # EX1 - LabelEncoder.classes_.size



# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx) #creates an object of type SentenceDataset and the constructor __init__ is executed
test_set = SentenceDataset(X_test, y_test, word2idx) 

"""
# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # EX7
"""


################## FOR EARLY STOPPING ###############
train_loader, val_loader = torch_train_val_split(
    dataset=train_set,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    val_size=0.2,
    shuffle=True,
    seed=42
)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

"""
model = BaselineDNN(output_size= n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

"""

"""
model = LSTM(output_size=n_classes,
             embeddings=embeddings,
             trainable_emb=EMB_TRAINABLE,
             bidirectional=True)

"""


"""
model = SimpleSelfAttentionModel(output_size=n_classes, embeddings=embeddings)
"""


"""
model = MultiHeadAttentionModel(output_size=n_classes, embeddings=embeddings)
"""

"""
model = TransformerEncoderModel(
    output_size=n_classes,
    embeddings=embeddings,
    max_length=35,       
    n_head=3,          
    n_layer=3           
).to(DEVICE)

"""

model = TransformerEncoderModel(
    output_size=n_classes,
    embeddings=embeddings,
    max_length=35,       
    n_head=8,          
    n_layer=6           
).to(DEVICE)




# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.CrossEntropyLoss()  # EX8
parameters = [param for param in model.parameters() if param.requires_grad]  # EX8
# param.requires_grad == True to avoid the parameters of the embeddings
optimizer = torch.optim.Adam(parameters, lr=0.001)  # EX8


#############################################################################
# Training Pipeline
#############################################################################
# lists to acummulate train and test losses 




"""
########################## Baseline DNN ###################################


TRAIN_LOSS = []
TEST_LOSS = []
Y_VAL_GOLD = []
Y_VAL_PRED = []




for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    TRAIN_LOSS.append(train_loss)
    TEST_LOSS.append(test_loss)

    # concatenate the predictions and gold labels in lists.
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )
    
    Y_TEST_GOLD.append(y_test_true)
    Y_TEST_PRED.append(y_test_pred)
    
    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", test_loss)
    print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_test_true, y_test_pred))
    print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_test_true, y_test_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_test_true, y_test_pred, average='macro'))




# plot training and validation loss curves
plt.plot(range(1, EPOCHS + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), TEST_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#keep the best metrics (best = epoch with minimum test_loss)
best_epoch = np.argmin(TEST_LOSS)
print(f"\nBest epoch: {best_epoch + 1}")
print("Accuracy:", accuracy_score(Y_TEST_GOLD[best_epoch], Y_TEST_PRED[best_epoch]))
print("F1:", f1_score(Y_TEST_GOLD[best_epoch], Y_TEST_PRED[best_epoch], average='macro'))
print("Recall:", recall_score(Y_TEST_GOLD[best_epoch], Y_TEST_PRED[best_epoch], average='macro'))




#save model
MODEL_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "MR_LSTM_model.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Μοντέλο αποθηκεύτηκε στο: {MODEL_PATH}")
"""





"""
##################### FOR LSTM #############################33

TRAIN_LOSS = []
VAL_LOSS = []
Y_VAL_GOLD = []
Y_VAL_PRED = []

MODEL_SAVE_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "Semival_LSTM_model_biderectional.pth")
early_stopper = EarlyStopper(model=model, save_path=MODEL_SAVE_PATH, patience=5, min_delta=0.0)


for epoch in range(1, EPOCHS + 1):
    # --- Εκπαίδευση ---
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # --- Training Metrics ---
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    y_train_true = np.concatenate(y_train_gold, axis=0)
    y_train_pred = np.concatenate(y_train_pred, axis=0)

    # --- Validation Metrics ---
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, criterion)
    y_val_true = np.concatenate(y_val_gold, axis=0)
    y_val_pred = np.concatenate(y_val_pred, axis=0)

    TRAIN_LOSS.append(train_loss)
    VAL_LOSS.append(val_loss)
    Y_VAL_GOLD.append(y_val_true)
    Y_VAL_PRED.append(y_val_pred)


    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", val_loss)
    print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_val_true, y_val_pred))
    print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_val_true, y_val_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_val_true, y_val_pred, average='macro'))


    if early_stopper.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# plot training and validation loss curves
plt.plot(range(1, len(TRAIN_LOSS) + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, len(VAL_LOSS) + 1), VAL_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



# --- Φόρτωση του καλύτερου μοντέλου (αποθηκεύτηκε από το EarlyStopper) ---
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()

# --- Τελική αξιολόγηση στο test set ---
final_test_loss, (final_test_gold, final_test_pred) = eval_dataset(test_loader, model, criterion)
final_y_true = np.concatenate(final_test_gold, axis=0)
final_y_pred = np.concatenate(final_test_pred, axis=0)

print("\nFinal Evaluation (Test Set):")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Accuracy: {accuracy_score(final_y_true, final_y_pred):.4f}")
print(f"F1 Score (macro): {f1_score(final_y_true, final_y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(final_y_true, final_y_pred, average='macro'):.4f}")

"""

















"""
##################### FOR ΑΤΤΕΝΤΙΟΝ #############################


TRAIN_LOSS = []
VAL_LOSS = []
Y_VAL_GOLD = []
Y_VAL_PRED = []

MODEL_SAVE_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "Semeval_Attention_model.pth")
early_stopper = EarlyStopper(model=model, save_path=MODEL_SAVE_PATH, patience=5, min_delta=0.0)


for epoch in range(1, EPOCHS + 1):
    # --- Εκπαίδευση ---
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # --- Training Metrics ---
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    y_train_true = np.concatenate(y_train_gold, axis=0)
    y_train_pred = np.concatenate(y_train_pred, axis=0)

    # --- Validation Metrics ---
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, criterion)
    y_val_true = np.concatenate(y_val_gold, axis=0)
    y_val_pred = np.concatenate(y_val_pred, axis=0)

    TRAIN_LOSS.append(train_loss)
    VAL_LOSS.append(val_loss)
    Y_VAL_GOLD.append(y_val_true)
    Y_VAL_PRED.append(y_val_pred)


    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", val_loss)
    print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_val_true, y_val_pred))
    print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_val_true, y_val_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_val_true, y_val_pred, average='macro'))


    if early_stopper.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# plot training and validation loss curves
plt.plot(range(1, len(TRAIN_LOSS) + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, len(VAL_LOSS) + 1), VAL_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



# --- Φόρτωση του καλύτερου μοντέλου (αποθηκεύτηκε από το EarlyStopper) ---
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()

# --- Τελική αξιολόγηση στο test set ---
final_test_loss, (final_test_gold, final_test_pred) = eval_dataset(test_loader, model, criterion)
final_y_true = np.concatenate(final_test_gold, axis=0)
final_y_pred = np.concatenate(final_test_pred, axis=0)

print("\nFinal Evaluation (Test Set):")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Accuracy: {accuracy_score(final_y_true, final_y_pred):.4f}")
print(f"F1 Score (macro): {f1_score(final_y_true, final_y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(final_y_true, final_y_pred, average='macro'):.4f}")


"""


















"""

##################### FOR MULTIHEAD ΑΤΤΕΝΤΙΟΝ #############################


TRAIN_LOSS = []
VAL_LOSS = []
Y_VAL_GOLD = []
Y_VAL_PRED = []

MODEL_SAVE_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "MR_Multihead_Attention_model.pth")
early_stopper = EarlyStopper(model=model, save_path=MODEL_SAVE_PATH, patience=5, min_delta=0.0)


for epoch in range(1, EPOCHS + 1):
    # --- Εκπαίδευση ---
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # --- Training Metrics ---
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    y_train_true = np.concatenate(y_train_gold, axis=0)
    y_train_pred = np.concatenate(y_train_pred, axis=0)

    # --- Validation Metrics ---
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, criterion)
    y_val_true = np.concatenate(y_val_gold, axis=0)
    y_val_pred = np.concatenate(y_val_pred, axis=0)

    TRAIN_LOSS.append(train_loss)
    VAL_LOSS.append(val_loss)
    Y_VAL_GOLD.append(y_val_true)
    Y_VAL_PRED.append(y_val_pred)


    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", val_loss)
    print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_val_true, y_val_pred))
    print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_val_true, y_val_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_val_true, y_val_pred, average='macro'))


    if early_stopper.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# plot training and validation loss curves
plt.plot(range(1, len(TRAIN_LOSS) + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, len(VAL_LOSS) + 1), VAL_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



# --- Φόρτωση του καλύτερου μοντέλου (αποθηκεύτηκε από το EarlyStopper) ---
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()

# --- Τελική αξιολόγηση στο test set ---
final_test_loss, (final_test_gold, final_test_pred) = eval_dataset(test_loader, model, criterion)
final_y_true = np.concatenate(final_test_gold, axis=0)
final_y_pred = np.concatenate(final_test_pred, axis=0)

print("\nFinal Evaluation (Test Set):")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Accuracy: {accuracy_score(final_y_true, final_y_pred):.4f}")
print(f"F1 Score (macro): {f1_score(final_y_true, final_y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(final_y_true, final_y_pred, average='macro'):.4f}")

"""






















##################### For Transformer #############################


TRAIN_LOSS = []
VAL_LOSS = []
Y_VAL_GOLD = []
Y_VAL_PRED = []

MODEL_SAVE_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "Semeval_Transformer_model_hyperpar.pth")
early_stopper = EarlyStopper(model=model, save_path=MODEL_SAVE_PATH, patience=5, min_delta=0.0)


for epoch in range(1, EPOCHS + 1):
    # --- Εκπαίδευση ---
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # --- Training Metrics ---
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    y_train_true = np.concatenate(y_train_gold, axis=0)
    y_train_pred = np.concatenate(y_train_pred, axis=0)

    # --- Validation Metrics ---
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, criterion)
    y_val_true = np.concatenate(y_val_gold, axis=0)
    y_val_pred = np.concatenate(y_val_pred, axis=0)

    TRAIN_LOSS.append(train_loss)
    VAL_LOSS.append(val_loss)
    Y_VAL_GOLD.append(y_val_true)
    Y_VAL_PRED.append(y_val_pred)


    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", val_loss)
    print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_val_true, y_val_pred))
    print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_val_true, y_val_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_val_true, y_val_pred, average='macro'))


    if early_stopper.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# plot training and validation loss curves
plt.plot(range(1, len(TRAIN_LOSS) + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, len(VAL_LOSS) + 1), VAL_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



# --- Φόρτωση του καλύτερου μοντέλου (αποθηκεύτηκε από το EarlyStopper) ---
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()

# --- Τελική αξιολόγηση στο test set ---
final_test_loss, (final_test_gold, final_test_pred) = eval_dataset(test_loader, model, criterion)
final_y_true = np.concatenate(final_test_gold, axis=0)
final_y_pred = np.concatenate(final_test_pred, axis=0)

print("\nFinal Evaluation (Test Set):")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Accuracy: {accuracy_score(final_y_true, final_y_pred):.4f}")
print(f"F1 Score (macro): {f1_score(final_y_true, final_y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(final_y_true, final_y_pred, average='macro'):.4f}")



















""" 
#MODEL LOADING

# Δημιουργείς το ίδιο μοντέλο όπως και στο training
model = BaselineDNN(output_size=n_classes, embeddings=embeddings, trainable_emb=EMB_TRAINABLE)
MODEL_PATH = os.path.join(r"C:\git_repo_nlp\slp-labs\lab3\saved_models", "baseline_model.pth")

# Φορτώνεις τα weights
model.load_state_dict(torch.load(MODEL_PATH))

# Στέλνεις σε σωστή συσκευή και θέτεις σε eval mode
model.to(DEVICE)
model.eval()
"""






########################################################  PROPARASKEYASTIKH ERGASIA ##############################################################

"""
#ZHTOYMENO 1


# Δείξε και τις αντιστοιχίες όλων των κλάσεων
print("\nMapping κατηγορίας → αριθμού:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{cls} → {i}")

# Πάρε τα πρώτα 10 labels (όπως ήταν αρχικά, σε string μορφή)
print("\nOriginal labels (πριν το encoding):")
print(y_train[:10])


print("\n So the corresponding list with the actual values is:")
print(y_train_orig[:10])

#lets print one neutral, one postive and one negative tweet
print("\nLets print one neutral, one postive and one negative tweet:")
print(f"Neutral:\n {X_train[0]}")
print(f"Positive:\n {X_train[1]}")
print(f"Negative:\n {X_train[8]}")

"""












"""
#ZHTOYMENO 2 
#self.tokenized_data is a list of lists that contain the tokenized sentences
print("\nLets print some tokenized examples:\n")
for i in range (10):
    print(f"Tokenized data {i}: {train_set.tokenized_data[i]}\n")
"""










"""
#ZHTOYMENO 3
print("\nLets print 5 examples of the encoded sentnces:\n")
for i in range (5):
    encoded, label, length = train_set.__getitem__(i)
    print(f"\nEncoded example: {encoded}")
    print(f"Example's label: {label}")
    print(f"Example's length: {length}") #length is the actual length before the zero padding
"""





