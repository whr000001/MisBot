import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Sampler, TensorDataset


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(3, 256),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(20, 256),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        x1, x3, x4, x5, y = data
        x1, x2 = torch.split(x1, [3, 20], dim=-1)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)
        x5 = self.fc5(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=-1)
        x = self.dropout(x)
        pred = self.classifier(x)

        loss = self.loss_fn(pred, y)
        return loss, pred


class MLPClassifier:
    def __init__(self, features, name, description, tweet, labels, device):
        self.model = MyModel().to(device)
        length = 48536
        train_size = int(length * 0.8)
        validation_size = int(length * 0.1)
        indices = np.arange(length)
        train_indices = indices[:train_size]
        validation_indices = indices[train_size:train_size + validation_size]
        test_indices = indices[train_size + validation_size:]
        train_sampler = MySampler(train_indices, shuffle=True)
        validation_sampler = MySampler(validation_indices, shuffle=False)
        test_sampler = MySampler(test_indices, shuffle=False)

        dataset = TensorDataset(features, name, description, tweet, labels)

        self.train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
        self.validation_loader = DataLoader(dataset, batch_size=64, sampler=validation_sampler)
        self.test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def train_one_epoch(self):
        self.model.train()
        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss, _ = self.model(data)
            loss.backward()
            self.optimizer.step()

    def validation(self):
        self.model.eval()
        all_truth = []
        all_preds = []
        for data in self.validation_loader:
            _, out = self.model(data)
            preds = out.argmax(-1).to('cpu')
            truth = data[-1].to('cpu')
            all_truth.append(truth)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_truth = torch.cat(all_truth, dim=0).numpy()
        return accuracy_score(all_truth, all_preds) * 100

    def test(self):
        self.model.eval()
        all_truth = []
        all_preds = []
        for data in self.test_loader:
            _, out = self.model(data)
            preds = out.argmax(-1).to('cpu')
            truth = data[-1].to('cpu')
            all_truth.append(truth)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_truth = torch.cat(all_truth, dim=0).numpy()
        metrics(all_truth, all_preds)

    def train(self):
        no_up_limit = 8
        no_up = 0

        best_acc = 0
        best_state = self.model.state_dict()
        for key, value in best_state.items():
            best_state[key] = value.clone()

        for _ in range(100):
            self.train_one_epoch()
            acc = self.validation()
            if acc > best_acc:
                best_acc = acc
                for key, value in self.model.state_dict().items():
                    best_state[key] = value.clone()
                no_up = 0
            else:
                no_up += 1
            if no_up >= no_up_limit:
                break
        self.model.load_state_dict(best_state)
        self.test()

    @torch.no_grad()
    def save(self, name):
        self.model.eval()
        all_truth = []
        all_preds = []
        all_logits = []
        for data in self.validation_loader:
            _, out = self.model(data)
            preds = out.argmax(-1).to('cpu')
            truth = data[-1].to('cpu')
            all_truth.append(truth)
            all_preds.append(preds)
            all_logits.append(out.to('cpu'))
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_truth = torch.cat(all_truth, dim=0).numpy()
        all_logits = torch.cat(all_logits, dim=0).numpy()
        torch.save([all_truth, all_preds, all_logits], f'{name}_validation.pt')

        all_truth = []
        all_preds = []
        all_logits = []
        for data in self.test_loader:
            _, out = self.model(data)
            preds = out.argmax(-1).to('cpu')
            truth = data[-1].to('cpu')
            all_truth.append(truth)
            all_preds.append(preds)
            all_logits.append(out.to('cpu'))
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_truth = torch.cat(all_truth, dim=0).numpy()
        all_logits = torch.cat(all_logits, dim=0).numpy()
        torch.save([all_truth, all_preds, all_logits], f'{name}_test.pt')


def metrics(truth, preds):
    print('${:.1f}$'.format(accuracy_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(f1_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(precision_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(recall_score(truth, preds) * 100))
    print('--------------------')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for lm in ['bert', 'deberta']:
        features = torch.load('../feature/features.pt', weights_only=True).to(device)
        name = torch.load(f'../content/{lm}_name.pt', weights_only=True).to(device)
        description = torch.load(f'../content/{lm}_description.pt', weights_only=True).to(device)
        tweet = torch.load(f'../content/{lm}_tweet.pt', weights_only=True).to(device)
        labels = torch.load('../content/labels.pt', weights_only=True).to(device)

        mlp = MLPClassifier(features, name, description, tweet, labels, device)
        mlp.train()
        mlp.save(f'ensemble_{lm}')


if __name__ == '__main__':
    main()
