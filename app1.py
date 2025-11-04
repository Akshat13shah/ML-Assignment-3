import streamlit as st
st.title("Loss Visualization")

import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import time
import matplotlib.pyplot as plt # for making figures
import pandas as pd
import numpy as np
import re



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("D:\IIT GANDHINAGAR\Semester 5\ES335 Machine Learning\Assignment3\shakespeare_input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = text.lower()
    text = re.sub('["\n\n"]', "   ", text)
    text = re.sub('[^a-z0-9 \.]', '', text)


text = text.strip()
text = text.split("   ")
text = [i for i in text if i != '']
#print(text[:500])
text = [i.replace(".","") for i in text]

#cxt = st.slider("Context Length :", 4,10)
op = st.selectbox("Activation, Embedding and Context", ["ReLU, 32, 5", "tanh, 64, 4"])

if op == "ReLU, 32, 5":
    activ = "ReLU"
    cxt = 5
    emb_d = 32
elif op == "tanh, 64, 4":
    activ = "tanh"
    cxt = 4
    emb_d = 64


X1, y1 = [],[]
#k,n = 0,0
for i in text:
#  cxt = 5
  out = cxt*["."]
  X1.append(out.copy())
  y1.append(i.strip().split()[0])
  for j in range (len(i.strip().split())-1):
    #print(out[1:4])
    out[0:cxt-1] = out[1:cxt]
    #print(out[0:3])
    #print(i,j)
    out[cxt-1] = i.strip().split()[j]
    
    #print(out[1:4])
    #Why? copy()
    X1.append(out.copy())
    y1.append(i.strip().split()[j+1])


X,X_test,y,y_test = [],[],[],[]
k = 0
for i in range (len(X1)):
  if X1[i] == [".",".",".",".","."]:
    k += 1
  if k < 13600:
    X.append(X1[i])
    y.append(y1[i])
  else:
    X_test.append(X1[i])
    y_test.append(y1[i])



word = sorted(list(set(y1)))

w_ = ["."]
w_.extend(word)

#emb_d = st.selectbox("Embedding dimension :", [16, 32, 64])
emb_l = len(word)
emb = torch.nn.Embedding(emb_l,emb_d)



class nxt_word(nn.Module):
  def __init__(self, cxt, emb_l, emb_d, size, activ):
    super().__init__()
    self.activ = activ
    self.emb = nn.Embedding(emb_l, emb_d)
    self.lin1 = nn.Linear(cxt * emb_d, size)
    self.lin2 = nn.Linear(size, emb_l)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    if self.activ == "tanh":
      x = torch.tanh(self.lin1(x))
    else:
      x = torch.relu(self.lin1(x))
    x = self.lin2(x)
    return x
  

def generate_name(model, word, inp , cxt,temp, max_len=10):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
 #       context = torch.tensor((5-len(inp))*[0]).to(device)+inp
 #       torch.cat((torch.tensor((5)*[0]).to(device),y_1))
        context = torch.cat((torch.tensor((cxt-len(inp))*[0]).to(device),inp))
        name = ''
        for i in range(max_len):
            x = torch.tensor(context).view(1, -1).to(device)
            y_pred = model(x)/temp
            ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
            ch = word[ix]
            if ch == '...':
                break
            name += " " + ch
            context = torch.cat((context[1:] , torch.tensor([ix]).to(device)))
    # Set model back to training mode
    model.train()
    return name



stoi = {w:i for i,w in enumerate(w_)}
itos = {i:w for w,i in stoi.items()}

X_ = [[stoi[i] for i in j] for j in X]
X_ = torch.tensor(X_).to(device)

y_ = [stoi[j] for j in y]
y_ = torch.tensor(y_).to(device)

X_p = [[stoi[i] for i in j] for j in X_test]
X_p = torch.tensor(X_p).to(device)

y_p = [stoi[j] for j in y_test]
y_p = torch.tensor(y_p).to(device)



if op == "ReLU, 32, 5":
    model = nxt_word(cxt, emb_l+1, emb_d, 1024, activ).to(device)
    #model.load_state_dict(torch.load("D:\IIT GANDHINAGAR\Semester 5\ES335 Machine Learning\Assignment3\model1.pth"))


    state_dict = torch.load("D:\IIT GANDHINAGAR\Semester 5\ES335 Machine Learning\Assignment3\model1.pth", map_location=device)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)


elif op == "tanh, 64, 4":
    model = nxt_word(cxt, emb_l+1, emb_d, 1024, activ).to(device)
#    model.load_state_dict(torch.load("D:\IIT GANDHINAGAR\Semester 5\ES335 Machine Learning\Assignment3\model11_weights.pth"))
    state_dict = torch.load("D:\IIT GANDHINAGAR\Semester 5\ES335 Machine Learning\Assignment3\model11_weights.pth", map_location=device)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)



#model = nxt_word(cxt, emb_l+1, emb_d, 1024, activ).to(device)
#model = torch.compile(model)



#inp = []
#n = int(input("Input length:"))
#for i in range (n):
#  inp.append(str(input()))

n = st.slider("Context length you give  :", 1,4,1)

# Create dynamic input boxes
inp = []
for i in range(int(n)):
    value = st.text_input(f"Enter word {i+1}:")
    if value:
        inp.append(value)


y_1 = [stoi[j] for j in inp]
y_1 = torch.tensor(y_1, dtype=torch.long, device=device)


temp = st.slider("Temperature :", 1,15,1)
st.write(generate_name(model, w_,y_1,cxt, temp))