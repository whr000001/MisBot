import json
import os.path
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader


class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', local_files_only=True)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

    def forward(self, data):
        out = self.bert(input_ids=data['input_ids'],
                        attention_mask=data['attention_mask'],
                        token_type_ids=data['token_type_ids'])
        reps = out.last_hidden_state
        attention_mask = data['attention_mask']
        reps = torch.einsum('ijk,ij->ijk', reps, attention_mask)
        reps = torch.sum(reps, dim=1)
        attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        reps = reps / attention_mask
        return reps


class NewsDataset(Dataset):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
        self.max_length = 256
        if os.path.exists('cache.pt'):
            self.data = torch.load('cache.pt', weights_only=True)
        else:
            data = json.load(open('dataset/main_data.json'))
            self.data = []
            for index, item in enumerate(tqdm(data)):
                content = self.tokenize(item['content'])
                repost_graph = item['repost_graph']
                comment_graphs = item['comment_graphs']
                repost_graph[0] = [self.tokenize(_) for _ in repost_graph[0]]
                for graph in comment_graphs:
                    graph[0] = [self.tokenize(_) for _ in graph[0]]
                self.data.append({
                    'content': content,
                    'repost_graph': repost_graph,
                    'comment_graphs': comment_graphs,
                })
            torch.save(self.data, 'cache.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def tokenize(self, data):
        input_ids = self.tokenizer.tokenize(data)
        input_ids = input_ids[:self.max_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids

    def get_collate_fn(self, device):
        def padding(data):
            max_length = 0
            for item in data:
                max_length = max(max_length, len(item))
            input_ids = []
            seq_lens = []
            attention_mask = []
            token_type_ids = []
            for item in data:
                input_ids.append(item + [self.tokenizer.pad_token_id] * (max_length - len(item)))
                attention_mask.append([1] * len(item) + [0] * (max_length - len(item)))
                token_type_ids.append([0] * max_length)
                seq_lens.append(len(item))
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long).to(device),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long).to(device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device),
                'seq_lens': torch.tensor(seq_lens, dtype=torch.long)
            }

        def collate_fn(batch):
            content = []

            repost = []
            repost_edge = []
            repost_batch = []

            comment = []
            comment_edge = []
            comment_batch = []
            comment_graph_batch = []

            graph_cnt = 0
            for index, item in enumerate(batch):
                content.append(item['content'])
                for repost_node in item['repost_graph'][0][1:]:
                    repost.append(repost_node)
                    repost_batch.append(index)
                repost_edge.append(torch.tensor(item['repost_graph'][1], dtype=torch.long).to(device))
                for graph in item['comment_graphs']:
                    comment_batch.append(index)
                    for node in graph[0]:
                        comment_graph_batch.append(graph_cnt)
                        comment.append(node)
                    comment_edge.append(torch.tensor(graph[1], dtype=torch.long).to(device))
                    graph_cnt += 1
            content = padding(content)
            repost = padding(repost)
            comment = padding(comment)
            repost_batch = torch.tensor(repost_batch, dtype=torch.long).to(device)
            comment_batch = torch.tensor(comment_batch, dtype=torch.long).to(device)
            comment_graph_batch = torch.tensor(comment_graph_batch, dtype=torch.long).to(device)
            return {
                'content': content,
                'repost': repost,
                'comment': comment,
                'repost_batch': repost_batch,
                'repost_edge': repost_edge,
                'comment_batch': comment_batch,
                'comment_graph_batch': comment_graph_batch,
                'comment_graph_cnt': graph_cnt,
                'comment_edge': comment_edge,
                'batch_size': len(batch)
            }
        return collate_fn


@torch.no_grad()
def main():
    dataset = NewsDataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.get_collate_fn(device), shuffle=False)
    model = BERTEncoder().to(device)
    model.eval()
    res = []
    for data in tqdm(loader):
        content = model(data['content'])
        if len(data['repost']['input_ids']) == 0:
            repost_x = torch.tensor([])
            repost_edge = torch.tensor([[], []])
        else:
            repost_x = model(data['repost'])
            repost_edge = data['repost_edge'][0]
        if data['comment_graph_cnt'] == 0:
            comment_x = torch.tensor([])
            comment_edge = torch.tensor([[], []])
        else:
            comment_x = model(data['comment'])
            comment_edge = data['comment_edge']
        res.append({
            'content': content,
            'repost_x': repost_x,
            'repost_edge': repost_edge,
            'comment_x': comment_x,
            'comment_edge': comment_edge
        })
    torch.save(res, 'dataset/main_data.pt')


if __name__ == '__main__':
    main()
