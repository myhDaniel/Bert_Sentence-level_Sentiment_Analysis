import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


model_name = 'bert-base-chinese'
TRAINSET_SIZE = 2500
TESTSET_SIZE = 500

with open('../data/clothing_comment/negdata.txt', 'r', encoding='utf-8') as f:
    neg_data = f.read()
with open('../data/clothing_comment/posdata.txt','r', encoding='utf-8') as f:
    pos_data = f.read()

neg_datalist = neg_data.split('\n')
pos_datalist = pos_data.split('\n')

dataset = np.array(pos_datalist + neg_datalist)
labels = np.array([1]*len(pos_datalist) + [0]*len(neg_datalist))

np.random.seed(10)
mix_index = np.random.choice(3000,3000)
dataset = dataset[mix_index]
labels = labels[mix_index]

train_samples = dataset[:TRAINSET_SIZE]
train_labels = labels[:TRAINSET_SIZE]
test_samples = dataset[TRAINSET_SIZE:TRAINSET_SIZE+TESTSET_SIZE]
test_labels = labels[TRAINSET_SIZE:TRAINSET_SIZE+TESTSET_SIZE]

def oneHot(l,size=2):
    res = list()
    for i in l:
        tmp = [0]*size
        tmp[i] = 1
        res.append(tmp)
    return res

tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_text = [tokenizer.tokenize(i) for i in train_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = oneHot(train_labels)

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) != 512:
        input_ids[j].extend([0]*(512-len(i)))

train_set = TensorDataset(torch.LongTensor(input_ids),torch.FloatTensor(input_labels))
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)

tokenized_text = [tokenizer.tokenize(i) for i in test_samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
input_labels = test_labels

for j in range(len(input_ids)):
    i = input_ids[j]
    if len(i) != 512:
        input_ids[j].extend([0]*(512-len(i)))

test_set = TensorDataset(torch.LongTensor(input_ids), torch.FloatTensor(input_labels))
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

class netWork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("./bert-base-chinese")
        for param in self.model.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.l = nn.Linear(768, 2)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[1]
        x = self.dropout(x)
        x = self.l(x)
        return x
    
net = netWork()
net.train()

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = optim.Adam(net.parameters(),lr=1e-5)

def predict(output):
    res = torch.argmax(output, 1)
    return res

accumulation_steps = 8
epoch = 3

for i in range(epoch):
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target.view(-1,2))

        mask = []
        for sample in data:
            mask.append([1 if i !=0 else 0 for i in sample])
        mask = torch.Tensor(mask)

        output = net(data, attention_mask=mask)
        pred = predict(output)

        loss = criterion(sigmoid(output).view(-1, 2), target)

        loss = loss/accumulation_steps
        loss.backward()

        if ((batch_idx+1) % accumulation_steps)==0:
            optimizer.step()
            optimizer.zero_grad()
        
        if ((batch_idx+1) % accumulation_steps) == 1:
            print(f"Train Epoch:{i+1}\tIteration:{batch_idx*4}/{TRAINSET_SIZE}\tLoss:{loss.item()}")
        
        if batch_idx == len(train_loader)-1:
            print("labels:",target)
            print("pred:",pred)

net.eval()

correct = 0
total = 0

for batch_idx, (data, target) in enumerate(test_loader):
    mask = []
    for sample in data:
        mask.append([1 if i != 0 else 0 for i in sample])
    mask = torch.Tensor(mask)

    output = net(data, attention_mask=mask)
    pred = predict(output)

    correct += (pred == target).sum().item()
    total += len(data)

print(f"Corret：{correct}, Total：{total}, Accuracy：{(100.0*correct)/total}%")