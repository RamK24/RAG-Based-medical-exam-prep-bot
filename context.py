import faiss
import torch
import numpy as np
from torch import Tensor
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


class Embeddings():

    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer    
        self.index = faiss.read_index("passages_7b.idx")
        self.doc_embeddings = torch.load('./embeddings_train.pth', weights_only=True)    


    def process_embeddings(self, embeddings):
        embeddings = embeddings.numpy().astype(np.float32)
        return embeddings
    
    def last_token_pool(self, last_hidden_states: Tensor,
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
    
    def tokenize(self, batch, max_length=512):
        passages = [f'Represent this passage\npassage: {content}' for content in batch["content"]]

        inputs = self.tokenizer(
            passages, 
            max_length=max_length - 1, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        batch_size = inputs['input_ids'].shape[0]
        eos_token_id = torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long)
        attention_val = torch.ones(batch_size, 1, dtype=torch.long)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], eos_token_id], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_val], dim=1)
        return inputs.to(self.model.device)
    

    def get_candidates(self, query):
        inputs = self.tokenize({'content': [query]})
        hidden_states = self.model(inputs['input_ids'].to(self.model.device), attention_mask=inputs['attention_mask'].to(self.model.device))
        query_embedding = self.last_token_pool(hidden_states.last_hidden_state, attention_mask=inputs['attention_mask'].to(self.model.device))
        query_embedding = self.process_embeddings(query_embedding.detach().cpu())
        distances, indices = self.index.search(query_embedding, 20)
        return distances, indices
    


    def get_context(self, query):
        _, indices = self.get_candidates(query)
        # context = ''
        # for i, idx in enumerate(indices.flatten().tolist()):
        #     context  += f'passage {i+1}: ' + docs[idx]['content'] + '\n\n'
        return indices

    

