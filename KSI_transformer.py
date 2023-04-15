import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import copy
from transformers import AutoModel, AutoTokenizer

##########################################################

label_to_ix=np.load('label_to_ix.npy', allow_pickle=True).item()
ix_to_label=np.load('ix_to_label.npy', allow_pickle=True)
training_data=np.load('training_data.npy', allow_pickle=True)
test_data=np.load('test_data.npy', allow_pickle=True)
val_data=np.load('val_data.npy', allow_pickle=True)
word_to_ix=np.load('word_to_ix.npy', allow_pickle=True).item()
ix_to_word=np.load('ix_to_word.npy', allow_pickle=True)
newwikivec=np.load('newwikivec.npy', allow_pickle=True)
wikivoc=np.load('wikivoc.npy', allow_pickle=True).item()



wikisize=newwikivec.shape[0]
rvocsize=newwikivec.shape[1]
wikivec=autograd.Variable(torch.FloatTensor(newwikivec))

batchsize=32



def preprocessing(data):

    new_data=[]
    for i, note, j in data:
        templabel=[0.0]*len(label_to_ix)
        for jj in j:
            if jj in wikivoc:
                templabel[label_to_ix[jj]]=1.0
        templabel=np.array(templabel,dtype=float)
        new_data.append((i, note, templabel))
    new_data=np.array(new_data)
    
    lenlist=[]
    for i in new_data:
        lenlist.append(len(i[0]))
    sortlen=sorted(range(len(lenlist)), key=lambda k: lenlist[k])  
    new_data=new_data[sortlen]
    
    batch_data=[]
    
    for start_ix in range(0, len(new_data)-batchsize+1, batchsize):
        thisblock=new_data[start_ix:start_ix+batchsize]
        mybsize= len(thisblock)
        numword=np.max([len(ii[0]) for ii in thisblock])
        main_matrix = np.zeros((mybsize, numword), dtype= np.int)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                try:
                    if thisblock[i][0][j] in word_to_ix:
                        main_matrix[i,j] = word_to_ix[thisblock[i][0][j]]
                    
                except IndexError:
                    pass       # because initialze with 0, so you pad with 0
    
        xxx2=[]
        yyy=[]
        for ii in thisblock:
            xxx2.append(ii[1])
            yyy.append(ii[2])
        
        xxx2=np.array(xxx2)
        yyy=np.array(yyy)
        batch_data.append((autograd.Variable(torch.from_numpy(main_matrix)),autograd.Variable(torch.FloatTensor(xxx2)),autograd.Variable(torch.FloatTensor(yyy))))
    return batch_data
batchtraining_data=preprocessing(training_data)
batchtest_data=preprocessing(test_data)
batchval_data=preprocessing(val_data)




######################################################################
# Create the model:

Embeddingsize=100
hidden_dim=200
class TransformerKSI(nn.Module):

    def __init__(self, batch_size, vocab_size, tagset_size, transformer_model_name):
        super(TransformerKSI, self).__init__()

        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.dropout = nn.Dropout(0.2)
        self.hidden2tag = nn.Linear(self.transformer.config.hidden_size, tagset_size)
        self.layer2 = nn.Linear(Embeddingsize, 1, bias=False)
        self.embedding = nn.Linear(rvocsize, Embeddingsize)
        self.vattention = nn.Linear(Embeddingsize, Embeddingsize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vec1, nvec, wiki, simlearning):
        # Tokenize input
        input_ids = vec1
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Pass input through the transformer
        output = self.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        # Apply dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # Compute tag scores
        vec2 = self.hidden2tag(last_hidden_state[:, 0, :])
        tag_scores = self.sigmoid(vec2)

        return tag_scores

######################################################################
# Train the model:

topk=10

def trainmodel(model, sim):
    print ('start_training')
    modelsaved=[]
    modelperform=[]
    topk=10
    
    
    bestresults=-1
    bestiter=-1
    for epoch in range(5000):  
        model.train()
        
        lossestrain = []
        recall=[]
        for mysentence in batchtraining_data:
            model.zero_grad()
            
            targets = mysentence[2].cuda()
            tag_scores = model(mysentence[0].cuda(),mysentence[1].cuda(),wikivec.cuda(),sim)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            lossestrain.append(loss.data.mean())
        print (epoch)
        modelsaved.append(copy.deepcopy(model.state_dict()))
        print ("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        model.eval()
    
        recall=[]
        for inputs in batchval_data:
           
            targets = inputs[2].cuda()
            tag_scores = model(inputs[0].cuda(),inputs[1].cuda() ,wikivec.cuda(),sim)
    
            loss = loss_function(tag_scores, targets)
            
            targets=targets.data.cpu().numpy()
            tag_scores= tag_scores.data.cpu().numpy()
            
            
            for iii in range(0,len(tag_scores)):
                temp={}
                for iiii in range(0,len(tag_scores[iii])):
                    temp[iiii]=tag_scores[iii][iiii]
                temp1=[(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=True)]
                thistop=int(np.sum(targets[iii]))
                hit=0.0
                for ii in temp1[0:max(thistop,topk)]:
                    if targets[iii][ii[0]]==1.0:
                        hit=hit+1
                if thistop!=0:
                    recall.append(hit/thistop)
            
        print ('validation top-',topk, np.mean(recall))
        
        
        
        modelperform.append(np.mean(recall))
        if modelperform[-1]>bestresults:
            bestresults=modelperform[-1]
            bestiter=len(modelperform)-1
        
        if (len(modelperform)-bestiter)>5:
            print (modelperform,bestiter)
            return modelsaved[bestiter]
    
# Instantiate the model with the desired transformer model name
transformer_model_name = "distilbert-base-uncased"
model = TransformerKSI(batchsize, len(word_to_ix), len(label_to_ix), transformer_model_name)
model.cuda()

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

basemodel = trainmodel(model, 0)
torch.save(basemodel, 'Transformer_model')

model = TransformerKSI(batchsize, len(word_to_ix), len(label_to_ix), transformer_model_name)
model.cuda()
model.load_state_dict(basemodel)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
KSImodel = trainmodel(model, 1)
torch.save(KSImodel, 'KSI_Transformer_model')


def testmodel(modelstate, sim):
    model = CNN(batchsize, len(word_to_ix), len(label_to_ix))
    model.cuda()
    model.load_state_dict(modelstate)
    loss_function = nn.BCELoss()
    model.eval()
    recall=[]
    lossestest = []
    
    y_true=[]
    y_scores=[]
    
    
    for inputs in batchtest_data:
       
        targets = inputs[2].cuda()
        
        tag_scores = model(inputs[0].cuda(),inputs[1].cuda() ,wikivec.cuda(),sim)

        loss = loss_function(tag_scores, targets)
        
        targets=targets.data.cpu().numpy()
        tag_scores= tag_scores.data.cpu().numpy()
        
        
        lossestest.append(loss.data.mean())
        y_true.append(targets)
        y_scores.append(tag_scores)
        
        for iii in range(0,len(tag_scores)):
            temp={}
            for iiii in range(0,len(tag_scores[iii])):
                temp[iiii]=tag_scores[iii][iiii]
            temp1=[(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=True)]
            thistop=int(np.sum(targets[iii]))
            hit=0.0
            
            for ii in temp1[0:max(thistop,topk)]:
                if targets[iii][ii[0]]==1.0:
                    hit=hit+1
            if thistop!=0:
                recall.append(hit/thistop)
    y_true=np.concatenate(y_true,axis=0)
    y_scores=np.concatenate(y_scores,axis=0)
    y_true=y_true.T
    y_scores=y_scores.T
    temptrue=[]
    tempscores=[]
    for  col in range(0,len(y_true)):
        if np.sum(y_true[col])!=0:
            temptrue.append(y_true[col])
            tempscores.append(y_scores[col])
    temptrue=np.array(temptrue)
    tempscores=np.array(tempscores)
    y_true=temptrue.T
    y_scores=tempscores.T
    y_pred=(y_scores>0.5).astype(np.int)

    

    try:
        lossestest_cpu_numpy = lossestest.cpu().numpy()

        print ('test loss', np.mean(lossestest))
    except:
        print('cant print test loss')
    try:
        print ('top-',topk, np.mean(recall))
    except:
        print('cant print topk')
    print ('macro AUC', roc_auc_score(y_true, y_scores,average='macro'))
    print ('micro AUC', roc_auc_score(y_true, y_scores,average='micro'))
    print ('macro F1', f1_score(y_true, y_pred, average='macro')  )
    print ('micro F1', f1_score(y_true, y_pred, average='micro')  )

print('Transformer alone:')
testmodel(basemodel, 0)
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('KSI+Transformer:')
testmodel(KSImodel, 1)

