from transformers import AutoModel, AutoTokenizer
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import faiss
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

model_name = "BMRetriever/BMRetriever-7B"
device="cuda"

model = AutoModel.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding


def tokenize(batch, train=True, max_length=512):
    # Process a batch of samples
    passages = [f'Represent this passage\npassage: {content}' for content in batch["content"]]

    # Tokenize the passage batch
    inputs = tokenizer(
        passages, 
        max_length=max_length - 1, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    
    # Add EOS token and extend attention mask
    batch_size = inputs['input_ids'].shape[0]
    eos_token_id = torch.full((batch_size, 1), tokenizer.eos_token_id, dtype=torch.long)
    attention_val = torch.ones(batch_size, 1, dtype=torch.long)
    inputs['input_ids'] = torch.cat([inputs['input_ids'], eos_token_id], dim=1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_val], dim=1)

    titles = batch["title"]
    numerical_labels = [label2id[title] for title in titles]

    # Convert tensors to lists for compatibility with the datasets library
    inputs['input_ids'] = inputs['input_ids'].tolist()
    inputs['attention_mask'] = inputs['attention_mask'].tolist()

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": numerical_labels
    }

def save_embeddings(model, data_loader, split='train'):
    embeddings_all = torch.empty((0, 4097))
    model.eval()
    for idx, inputs in  enumerate(data_loader):
        with torch.no_grad():
            embeddings = model(inputs['input_ids'].to(model.device), attention_mask=inputs['attention_mask'].to(model.device))
            embeddings = last_token_pool(embeddings.last_hidden_state, attention_mask=inputs['attention_mask'].to(model.device))
            embeddings = torch.cat([embeddings.detach().cpu(), inputs['labels'].unsqueeze(1)], dim=1)
            embeddings_all = torch.cat([embeddings_all, embeddings], dim=0)
    torch.save(embeddings_all, f'embeddings_{split}.pth')



class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.fc2 = nn.Linear(2048, 18)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
        

def get_torch_dataset(split):
    ds = split.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

ds = load_dataset("MedRAG/textbooks")
labels = ds["train"]["title"]
unique_titles = np.unique(labels) 
label2id = {title: idx for idx, title in enumerate(unique_titles)}
id2label = {idx: title for title, idx in label2id.items()}

ds = load_dataset("MedRAG/textbooks")
train_ds = get_torch_dataset(ds)
train_dataloader = DataLoader(train_ds['train'], batch_size=4)
save_embeddings(model, train_dataloader)
