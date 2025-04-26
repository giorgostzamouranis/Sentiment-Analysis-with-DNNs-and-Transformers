import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # EX4
        # 1 - define the embedding layer      
        num_embeddings = len(embeddings) #ex. 400002 words
        dim = len(embeddings[0]) #columns of embedings (vectors dimension)
        dim_max_mean_pooling = 2*dim
        self.embeddings = nn.Embedding(num_embeddings, dim)

        """
        here we created an embedding layer, so each token (ex. token = 123) will be 
        matched with a vector of size 50
        """

        self.output_size = output_size #outputs size

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True) #replace embedding layer with the actual embeddings without changing the weigths (freeze)

        # 4 - define a non-linear transformation of the representations
        # EX5
        self.linear = nn.Linear(dim_max_mean_pooling, 1000) #fully connected layer with a hidden layer of dimension=1000
        self.relu = nn.ReLU()



        ############### drop out ###############
        #self.dropout = nn.Dropout(p=0.3)




        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.output = nn.Linear(1000, output_size) #get the 1000 input of the previous and convert it to logits





    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        # EX6
        embeddings = self.embeddings(x) #x=token ids and convert them to word vectors
                                        # output = [batch_size, max_length, emb_dim]
                                        # so for each tweet an array 35x50 because we have max 35 tokens


        # 2 - construct a sentence representation out of the word embeddings
        # EX6
        # calculate the means
        representations = torch.sum(embeddings, dim=1) #we sum over the 35 token vectors (dim=1) so representations.shape = [batch_size, emb_dim]
        for i in range(lengths.shape[0]): # necessary to skip zeros in mean calculation
            representations[i] = representations[i] / lengths[i] #each sentence is now an array of vectors so I want to take the mean of these vectors-words
        
        
        ##################### Question 1 ########################
        """
        We will make our representation as the concatenation of mean pooling
        and max pooling.

        Example
             E = [
            [0.1, 0.4, 0.9],
            [0.3, 0.6, 0.2],
            [0.5, 0.2, 0.3]
            ]
            mean = [ (0.1+0.3+0.5)/3, ... ] = [0.3, 0.4, 0.466]
            max = [ max(0.1,0.3,0.5), ... ] = [0.5, 0.6, 0.9]
            concatenation = [0.3, 0.4, 0.466, 0.5, 0.6, 0.9] 
        """
        
    # --- max pooling ---
        batch_size = embeddings.shape[0]
        emb_dim = embeddings.shape[2]
        max_pooled = torch.zeros((batch_size, emb_dim), device=embeddings.device)
        for i in range(batch_size):
            real_len = lengths[i]
            real_tokens = embeddings[i, :real_len]  #only real tokens not padding
            max_pooled[i] = real_tokens.max(dim=0).values
            
            
        # --- concatenate mean + max ---
        representations = torch.cat([representations, max_pooled], dim=1)  # shape: [batch_size, emb_dim * 2]

        

        # 3 - transform the representations to new ones.
        # EX6
        representations = self.relu(self.linear(representations)) #output = [batch_size, 1000]


        ############# drop out ##############3
        #representations = self.dropout(representations)

        # 4 - project the representations to classes using a linear layer
        # EX6
        logits = self.output(representations)

        return logits











class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100 #dimension of the output of the h()
        #input of h()=emb dimension but output=hideen_size
        self.num_layers = 1 #only one LSTM layer
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb: #we freeze the embeddings training if we have trainable=false
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size) #takes as input the final hn and convert it to output_size

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x) #tokens are converted to vectors shape = [batch_size, max_length, emb_dim]
       
        #fix: μετατρέπουμε και στέλνουμε στη σωστή συσκευή
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, dtype=torch.long)
        lengths = lengths.to(x.device)

        # Σιγουριά ότι τα lengths δεν ξεφεύγουν από max_len
        lengths = torch.clamp(lengths, max=embeddings.shape[1])
        
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False) #we skip the padded tokens and get only the actual length

        ht, _ = self.lstm(X) #the outputs h1,h2,h3,... 
        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True) #unpack so ht.shape=[batch_size, max_seq_len, hidden_size]

        """
        #pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        


        if we write ht[:,max_length, :] then we get for all the batches, the last word (even if it is padded), all the 100 dimensions
        
        We actually want to get the non padded length so the sentence i has lengths[i] real tokens so we need somthing like ht[i, lengths[i]-1, :]

        
        """
        if not self.bidirectional:
            idx = (lengths-1).view(-1, 1).expand(-1, ht.size(2))  # [batch_size, hidden_size]
            representations = ht.gather(1, idx.unsqueeze(1)).squeeze(1)  # [batch_size, hidden_size]


        else:
            #we get hn (last output of forward LSTM) and h0 (last output of backward LSTM)
            #ht has shape: [batch_size, max_len, 2 * hidden_size]
            forward_out = ht[range(batch_size), lengths - 1, :self.hidden_size] # :self.hidden_size keeps only the forward output
            backward_out = ht[:, 0, self.hidden_size:] #backward reads from the end to start so we have 0
            representations = torch.cat([forward_out, backward_out], dim=1)  # [batch_size, 2*hidden_size]


        logits = self.linear(representations)

        return logits
