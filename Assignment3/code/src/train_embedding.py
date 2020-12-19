import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os, sys
import pickle
from word2vec_models import CBOW, Skipgram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_context_vector(context, word_to_ix):
    idx = [word_to_ix[w] for w in context]
    return torch.tensor(idx, dtype = torch.long)

def train_epoch(trainloader, net, optimizer, criterion):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        context, target = data
        context.to(device)
        target.to(device)
        
        #set gradients to 0
        optimizer.zero_grad()
        
        #find loss
        outputs = net(context)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss
    
    
#Specify arguments for training
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", type=str, choices=['cbow', 'skipgram'], 
                    default='cbow',help='Specify training algorithm(CBOW or Skipgram). Defaults to cbow')
parser.add_argument("-d","--embedding-dim",type=int, default=100, 
                    help='Specify dimension of embedding. Defaults to 100')
parser.add_argument("-cs","--context-size",type=int, default=2, 
                    help='Specify size of context window. Defaults to 2')
parser.add_argument("-e","--epochs",type=int,default=5, 
                    help='No of epochs to train. Defaults to 5')
args = parser.parse_args()

#Load cleaned words and dictionary
with open('../data/cleaned_words.pickle', 'rb') as handle:
    corpus = pickle.load(handle)
with open('../data/word_to_ix.pickle', 'rb') as handle:
    word_to_ix = pickle.load(handle)
with open('../data/ix_to_word.pickle', 'rb') as handle:
    ix_to_word = pickle.load(handle)
vocab_size = len(word_to_ix)

# Specify model name for word2vec
model_name = args.model + '_d' + str(args.embedding_dim) + '_cs_'+str(args.context_size)

if args.model == 'cbow':
    print('Training CBOW model')    
    net = CBOW(vocab_size, args.embedding_dim, 2*args.context_size)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001)    

    print('Creating context-target pairs')
    # Create (context, target) pairs and save in trainloader
    train_data = []
    for i in tqdm(range(args.context_size,len(corpus) - args.context_size)):
        context = [corpus[i-j] for j in range(1,args.context_size+1)]
        context.extend([corpus[i+j] for j in range(1,args.context_size+1)])
        target = [corpus[i]]
        train_data.append((make_context_vector(context,word_to_ix),
                           make_context_vector(target,word_to_ix)))
    print(train_data[:5])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=2, batch_size=512)

    print('Starting Training')
    for epoch in range(args.epochs):
        
        epoch_loss = train_epoch(trainloader, net, optimizer, criterion)
        print('epoch: %d loss: %.4f'%(epoch+1,epoch_loss))
        
        #Checkpoint model every epoch
        model_path = '../models/'+str(model_name)+'_'+str(epoch+1)+'.pth'
        torch.save({'epoch':epoch+1,
                    'loss':epoch_loss,
                    'model_state_dict':net.state_dict() }, model_path)

else:
    print('Training skipgram model')
    net = Skipgram(vocab_size, args.embedding_dim, args.context_size)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001)    

    print('Creating context-target pairs')
    # Create (context, target) pairs and save in trainloader
    train_data = []
    for i in tqdm(range(args.context_size,len(corpus) - args.context_size)):
        target = [corpus[i-j] for j in range(1,args.context_size+1)]
        target.extend([corpus[i+j] for j in range(1,args.context_size+1)])
        context = [corpus[i]]
        train_data.append((make_context_vector(context,word_to_ix),
                           make_context_vector(target,word_to_ix)))
    print(train_data[:5])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=2, batch_size=512)

    print('Starting Training')
    for epoch in range(args.epochs):
        
        epoch_loss = train_epoch(trainloader, net, optimizer, criterion)
        print('epoch: %d loss: %.4f'%(epoch+1,epoch_loss))
        
        model_path = '../models/'+str(model_name)+'_'+str(epoch+1)+'.pth'
        torch.save({'epoch':epoch+1,
                    'loss':epoch_loss,
                    'model_state_dict':net.state_dict() }, model_path)

print("Finished training word2vec model")