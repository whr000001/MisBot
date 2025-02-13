import torch
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def metrics(truth, preds):
    print('${:.1f}$'.format(accuracy_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(f1_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(precision_score(truth, preds) * 100), end='&')
    print('${:.1f}$'.format(recall_score(truth, preds) * 100))
    print('--------------------')


def main():
    experts = ['feature/feature',
               'content/bert_name', 'content/deberta_name',
               'content/bert_description', 'content/deberta_description',
               'content/bert_tweet', 'content/deberta_tweet',
               'ensemble/ensemble_bert', 'ensemble/ensemble_deberta']
    temperatures = [1.125, 1.468, 1.408, 1.246, 0.972, 1.286, 1.129, 1.329, 1.146]
    all_truth = None
    all_preds = []
    for index in [5, 6, 7, 8]:
        expert = experts[index]
        temperature = temperatures[index]
        data = torch.load(f'{expert}_test.pt', weights_only=False)
        truth, _, logits = data
        truth = torch.tensor(truth)
        logits = torch.tensor(logits)
        likelihood = torch.softmax(logits / temperature, dim=-1)[:, 1]
        if all_truth is None:
            all_truth = truth
        all_preds.append(likelihood)
    all_preds = torch.stack(all_preds)
    all_preds = torch.mean(all_preds, dim=0)
    all_preds = torch.greater(all_preds, 0.75).numpy()
    all_truth = all_truth.numpy()
    metrics(all_truth, all_preds)


if __name__ == '__main__':
    main()
