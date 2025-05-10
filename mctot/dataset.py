import json
from torch.utils.data import Dataset


class HotpotQA(Dataset):
    def __init__(self, data_pth, mode):
        with open(data_pth, 'r') as f:
            self.dataset = json.load(f)
        
        assert mode in ['eval', 'database', 'all']
        self.mode = mode

        if self.mode == 'database':
            self.dataset = [doc['context'] for doc in self.dataset]
            self.dataset = [doc for docs in self.dataset for doc in docs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'eval':
            return {
                '_id': self.dataset[idx]['_id'],
                'question': self.dataset[idx]['question'],
                'answer': self.dataset[idx]['answer']
            }
        elif self.mode == 'database':
            context = self.dataset[idx][0] + ': '
            for text in self.dataset[idx][1]:
                context += text
            return context
        else:
            return self.dataset[idx]
        
class Bamboogle(Dataset):
    def __init__(self, data_pth, mode):
        with open(data_pth, 'r') as f:
            self.dataset = json.load(f)
        
        assert mode in ['eval', 'database', 'all']
        self.mode = mode

        if self.mode == 'database':
            data = self.dataset['data']
            part1 = [f"{d['Q1']} {d['A1'][0]}" for d in data if d.get('A1') and len(d['A1']) > 0]
            part2 = [f"{d['Q2']} {d['A2'][0]}" for d in data if d.get('A2') and len(d['A2']) > 0]
            combined = part1 + part2
            seen = set()
            self.dataset = []
            for item in combined:
                if item not in seen:
                    seen.add(item)
                    self.dataset.append(item)
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class WikiMultiHopQA(Dataset):
    def __init__(self, data_pth, mode):
        with open(data_pth, 'r') as f:
            self.dataset = json.load(f)
        
        assert mode in ['eval', 'database', 'all']
        self.mode = mode

        if self.mode == 'database':
            self.dataset = [doc['context'] for doc in self.dataset]
            self.dataset = [doc for docs in self.dataset for doc in docs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'eval':
            return {
                '_id': self.dataset[idx]['_id'],
                'question': self.dataset[idx]['question'],
                'answer': self.dataset[idx]['answer']
            }
        elif self.mode == 'database':
            context = self.dataset[idx][0] + ': '
            for text in self.dataset[idx][1]:
                context += text
            return context
        else:
            return self.dataset[idx]