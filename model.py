import torch
import torch.nn as nn
import torchvision.models as models


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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size) # The shape is number of vocabulary,  size of embedding vector of caption
        
       
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, #number of units in hidden LSTM layer
                             num_layers = num_layers,  #default number of LSTM layers =1
                             dropout = 0, 
                             batch_first=True
                           )
        self.linear = nn.Linear(hidden_size,vocab_size)
        
   
    
    def forward(self, features, captions):  
        # create embedded word vectors for each word in a caption except last word (Discard the <end> word)
        embeds = self.embed(captions[:,:-1]) # Pass image captions through the word_embeddings layer
        
        output_cat = torch.cat((features.unsqueeze(1), embeds),dim = 1) # Concatenate the feature vectors for image and captions
        # Features shape : (batch_size, embed_size)
        # Word embeddings (embeds) shape : (batch_size, caption length , embed_size)
        # output_cat shape : (batch_size, caption length, embed_size)
        
        lstm_output, hidden_layer = self.lstm(output_cat)
        #LSTM_output shape is (batch_size, caption length, hidden_size)
        
        return self.linear(lstm_output)  ## Linear layer output shape : (batch_size, caption length, vocab_size)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pred_caption = []
        for i in range(0,max_len):
            output, states = self.lstm(inputs, states)
            _ , pred_word_idx = torch.max(self.linear(output.squeeze(dim = 1)), 1)
            inputs = self.embed(pred_word_idx).unsqueeze(1)
            if pred_word_idx == 1: # Come out when <end> is found
                break
            if pred_word_idx != 0:  #Not print <start>
                pred_caption.append(int(pred_word_idx.cpu().numpy()[0]))  
            
        return pred_caption   
            
            
            