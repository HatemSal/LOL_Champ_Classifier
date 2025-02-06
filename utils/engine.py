from tqdm.auto import tqdm
import torch
from torch import nn

def train_model(model,train_dataloader,loss_fn,optimizer,EPOCHS):
  for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n------------")
    model.train()
    train_loss =0
    train_acc=0
    for X, y in train_dataloader:
      X, y = X.to(device), y.to(device)
      y_logits = model(X)
      loss = loss_fn(y_logits,y)
      train_loss +=loss
      y_preds = y_logits.argmax(dim=1)
      train_acc += ((y_preds==y).sum())/len(y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
