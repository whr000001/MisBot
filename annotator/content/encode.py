import os
import json
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from torch_scatter import scatter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


class MyDataset(Dataset):
    def __init__(self, attribute, lm):
        assert attribute in ['tweet', 'description', 'name']
        if lm == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
        elif lm == 'deberta':
            self.tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese')
        else:
            raise KeyError
        self.max_length = 256
        self.attribute = attribute
        self.data = []
        random.seed(20250101)
        data_path = '../datasets/Sampled/'
        files = sorted(os.listdir(data_path))
        random.shuffle(files)
        labels = []
        for file in tqdm(files):
            data = json.load(open(f'{data_path}/{file}'))
            labels.append(data['label'])
            if attribute == 'tweet':
                out_tweet = []
                for text in data[attribute]:
                    out_tweet.append(self._word_embedding(text))
                self.data.append({
                    'attribute': out_tweet,
                })
            else:
                self.data.append({
                    'attribute': self._word_embedding(data[attribute]),
                })
        labels = torch.tensor(labels, dtype=torch.long)
        torch.save(labels, 'labels.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _word_embedding(self, text):
        words = self.tokenizer.tokenize(text)
        words = words[:self.max_length - 2]
        words = [self.tokenizer.cls_token] + words + [self.tokenizer.sep_token]
        tokens = self.tokenizer.convert_tokens_to_ids(words)
        return tokens

    def get_collate_fn(self, device):
        def _get_batch(tensor_list):
            max_length = 0
            input_ids = []
            token_type_ids = []
            attention_mask = []
            for item in tensor_list:
                max_length = max(max_length, len(item))
            for item in tensor_list:
                input_ids.append(item + [self.tokenizer.pad_token_id] * (max_length - len(item)))
                token_type_ids.append([0] * max_length)
                attention_mask.append([1] * len(item) + [0] * (max_length - len(item)))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            return {
                'input_ids': input_ids.to(device),
                'token_type_ids': token_type_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }

        def collate_fn(batch):
            if self.attribute == 'tweet':
                batch_attribute = []
                data_batch = []
                for index, item in enumerate(batch):
                    batch_attribute += item['attribute']
                    data_batch += [index] * len(item['attribute'])
                batch_attribute = _get_batch(batch_attribute)
                data_batch = torch.tensor(data_batch, dtype=torch.long).to(device)
                return {
                    'data': batch_attribute,
                    'data_batch': data_batch,
                }
            else:
                batch_attribute = []
                for index, item in enumerate(batch):
                    batch_attribute.append(item['attribute'])
                batch_attribute = _get_batch(batch_attribute)
                return {
                    'data': batch_attribute,
                }

        return collate_fn


class MyModel(nn.Module):
    def __init__(self, attribute, lm):
        super().__init__()
        if lm == 'bert':
            self.lm = AutoModel.from_pretrained('google-bert/bert-base-chinese')
        elif lm == 'deberta':
            self.lm = AutoModel.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese')
        self.attribute = attribute

    def forward(self, data):
        if self.attribute == 'tweet':
            data_batch = data['data_batch']
            size = int(data_batch.max().item() + 1)
            out = self.lm(**data['data'])
            features = out.last_hidden_state
            attention_mask = data['data']['attention_mask']
            features = torch.einsum('ijk,ij->ijk', features, attention_mask)
            features = torch.sum(features, dim=1)
            attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            features = features / attention_mask
            features = scatter(features, data_batch, dim=-2, dim_size=size, reduce='mean')
        else:
            out = self.lm(**data['data'])
            features = out.last_hidden_state
            attention_mask = data['data']['attention_mask']
            features = torch.einsum('ijk,ij->ijk', features, attention_mask)
            features = torch.sum(features, dim=1)
            attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            features = features / attention_mask
        return features


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for attribute in ['name', 'description', 'tweet']:
        for lm in ['bert', 'deberta']:
            dataset = MyDataset(attribute, lm)
            model = MyModel(attribute, lm).to(device)
            loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dataset.get_collate_fn(device))
            reps = []
            for item in tqdm(loader):
                out = model(item)
                reps.append(out.to('cpu'))
            reps = torch.cat(reps, dim=0)
            torch.save(reps, f'{lm}_{attribute}.pt')


if __name__ == '__main__':
    main()

