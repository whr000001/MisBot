import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(23, 128),
            nn.LeakyReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.dropout(self.fc(x))
        pred = self.classifier(x)

        loss = self.loss_fn(pred, y)
        return loss, pred


class MLPClassifier:
    def __init__(self, train_x, train_y, validation_x, validation_y, test_x, test_y, device):
        self.model = MyModel().to(device)
        train_x = torch.tensor(train_x, dtype=torch.float).to(device)
        validation_x = torch.tensor(validation_x, dtype=torch.float).to(device)
        test_x = torch.tensor(test_x, dtype=torch.float).to(device)

        train_y = torch.tensor(train_y, dtype=torch.long).to(device)
        validation_y = torch.tensor(validation_y, dtype=torch.long).to(device)
        test_y = torch.tensor(test_y, dtype=torch.long).to(device)

        train_set = TensorDataset(train_x, train_y)
        validation_set = TensorDataset(validation_x, validation_y)
        test_set = TensorDataset(test_x, test_y)
        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        self.validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def train_one_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            loss, _ = self.model(x, y)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        all_truth = []
        all_preds = []
        for x, y in self.validation_loader:
            _, out = self.model(x, y)
            preds = out.argmax(-1).to('cpu')
            truth = y.to('cpu')
            all_truth.append(truth)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_truth = torch.cat(all_truth, dim=0).numpy()
        return accuracy_score(all_truth, all_preds) * 100

    @torch.no_grad()
    def test(self):
        self.model.eval()
        all_truth = []
        all_preds = []
        for x, y in self.test_loader:
            _, out = self.model(x, y)
            preds = out.argmax(-1).to('cpu')
            truth = y.to('cpu')
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
        for x, y in self.validation_loader:
            _, out = self.model(x, y)
            preds = out.argmax(-1).to('cpu')
            truth = y.to('cpu')
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
        for x, y in self.test_loader:
            _, out = self.model(x, y)
            preds = out.argmax(-1).to('cpu')
            truth = y.to('cpu')
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
    random.seed(20250101)
    data_path = '../datasets/Sampled/'
    files = sorted(os.listdir(data_path))
    random.shuffle(files)
    numerical = []
    categorical = []
    labels = []
    for file in tqdm(files):
        data = json.load(open(f'{data_path}/{file}'))
        numerical.append(data['numerical'])
        categorical.append(data['categorical'])
        labels.append(data['label'])
    numerical = np.array(numerical, dtype=float)
    categorical = np.array(categorical)
    numerical_mean = np.array([5074.88, 420.59, 1432.10], dtype=float)
    numerical_std = np.array([283145.61, 584.40, 1373.91], dtype=float)
    numerical = (numerical - numerical_mean) / numerical_std
    features = np.concatenate([numerical, categorical], axis=-1)
    features_torch = torch.tensor(features, dtype=torch.float)
    torch.save(features_torch, 'features.pt')
    labels = np.array(labels)
    train_size = int(len(files) * 0.8)
    validation_size = int(len(files) * 0.1)
    indices = np.arange(len(files))
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size+validation_size]
    test_indices = indices[train_size+validation_size:]
    train_x, train_y = features[train_indices], labels[train_indices]
    validation_x, validation_y = features[validation_indices], labels[validation_indices]
    test_x, test_y = features[test_indices], labels[test_indices]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = MLPClassifier(train_x, train_y, validation_x, validation_y, test_x, test_y, device)
    mlp.train()
    mlp.save('feature')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    y_pred = rf.predict(test_x)
    metrics(test_y, y_pred)

    base_estimator = DecisionTreeClassifier(random_state=42)
    adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42, algorithm='SAMME')
    adaboost.fit(train_x, train_y)
    y_pred = adaboost.predict(test_x)
    metrics(test_y, y_pred)


if __name__ == '__main__':
    main()
