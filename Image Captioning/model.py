import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        

        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, 
                            batch_first=True,
                            dropout=0,
                            bidirectional=False
                           )
       
        self.linear = nn.Linear(hidden_size, vocab_size)                     

        # initialize the hidden state
        # self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size):
        
        self.batch_size = batch_size
        
        return (torch.zeros((1, self.batch_size, self.hidden_size), device=device), \
                torch.zeros((1, self.batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        
        
        # Discard the <end> word 
        captions = captions[:, :-1]     
        
        # Initialize the hidden state
        self.batch_size = features.shape[0] 
        self.hidden = self.init_hidden(self.batch_size) 
                
        
        embeddings = self.word_embeddings(captions)
        
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) 

        return outputs

    
    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        
        output = []
        self.batch_size = inputs.shape[0] 
        hidden = self.init_hidden(self.batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.linear(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_indice = torch.max(outputs, dim=1) 
            
            output.append(max_indice.cpu().numpy()[0].item())
            
            if (max_indice == 1):
                # We predicted the <end> word,so break
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.word_embeddings(max_indice) 
            inputs = inputs.unsqueeze(1) 
            
        return output

