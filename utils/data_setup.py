import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

def create_train_dataloader(train_dir,transform,batch_size):
  train_dataset = datasets.ImageFolder(root=train_dir,transform=transform,target_transform=None)
  class_dict = train_dataset.class_to_idx
  class_names = train_dataset.classes
  train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  return train_dataloader, class_names, class_dict
