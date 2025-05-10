from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import faiss
from accelerate import Accelerator
from dataset import *


class Retriever:
    def __init__(self, model_pth, data_pth):
        model = SentenceTransformer(model_pth)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self.accelerator.prepare_model(model)
        if 'hotpot' in data_pth.lower():
            self.dataset = HotpotQA(data_pth=data_pth, mode='database')
        elif 'wiki' in data_pth.lower():
            self.dataset = WikiMultiHopQA(data_pth=data_pth, mode='database')
        elif 'bamboogle' in data_pth.lower():
            self.dataset = Bamboogle(data_pth=data_pth, mode='database')
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
    
    def build_index(self):
        # encode the whole dataset into embeddings
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=1)
        data_iter = tqdm.tqdm(enumerate(dataloader), desc=f"Text Encoding ...", total=len(dataloader), bar_format="{l_bar}{r_bar}")
        
        embeds = []
        for _, data in data_iter:
            embed = self.model.encode(data, normalize_embeddings=True)
            embeds.append(embed)

        embeds = np.concatenate(embeds, axis=0)
        self.index.train(embeds)
        self.index.add(embeds)

    def retrieve(self, query, k):
        q_embed = self.model.encode(query, normalize_embeddings=True)
        distances, indices = self.index.search(q_embed, k)
        
        return distances, indices
    
    def get_document(self, index):
        return self.dataset[index]

    def save_index(self, index_pth):
        faiss.write_index(self.index, index_pth)

    def load_index(self, index_pth):
        self.index = faiss.read_index(index_pth)


if __name__ == '__main__':
    retriever = Retriever(model_pth="model_pth", data_pth="mctot/data/wiki_dev.json")

    retriever.build_index()
    retriever.save_index(index_pth="index_pth")
