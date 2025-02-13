import torch
import math
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def calculate_ece(label, pred, likelihood):
    limits = [[_ * 0.1, _ * 0.1 + 0.1] for _ in range(10)]
    bins = [[] for _ in range(len(limits))]
    for item_pred, item_label, item_likelihood in zip(pred, label, likelihood):
        if item_likelihood <= 0.5:
            item_likelihood = 1 - item_likelihood
        bins_index = None
        for _, limit in enumerate(limits):
            if limit[0] <= item_likelihood <= limit[1]:
                bins_index = _
                break
        assert bins_index is not None
        bins[bins_index].append([item_pred, item_label, item_likelihood])
    bin_acc = []
    ece = 0.0
    bin_cnt = []
    for each_bin in bins:
        bin_pred, bin_label, bin_likelihood = [], [], []
        bin_cnt.append(len(each_bin))
        for item_pred, item_label, item_likelihood in each_bin:
            bin_pred.append(item_pred)
            bin_label.append(item_label)
            bin_likelihood.append(item_likelihood)
        if len(bin_pred) == 0:
            bin_acc.append(None)
            continue
        bin_acc.append(accuracy_score(bin_label, bin_pred))
        ece += len(each_bin) / len(pred) * math.fabs(bin_acc[-1] - np.mean(bin_likelihood))
    return bin_cnt, bin_acc, ece


def calibration(data):
    truth, preds, logits = data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    truth = torch.tensor(truth).to(device)
    preds = torch.tensor(preds).to(device)
    logits = torch.tensor(logits).to(device)
    likelihood = torch.softmax(logits, dim=-1)[:, 1]
    _, _, ece = calculate_ece(truth.to('cpu').numpy(),
                              preds.to('cpu').numpy(),
                              likelihood.to('cpu').numpy())
    print('initial ece: ', ece)
    min_ece = None
    best_temperature = None
    for temperature in tqdm(np.arange(0.5, 1.5, 0.001), leave=False):
        likelihood = torch.softmax(logits / temperature, dim=-1)[:, 1]
        _, _, ece = calculate_ece(truth.to('cpu').numpy(),
                                  preds.to('cpu').numpy(),
                                  likelihood.to('cpu').numpy())
        if best_temperature is None or ece < min_ece:
            min_ece = ece
            best_temperature = temperature
    # print(min_ece)
    print('best temperature: ', best_temperature)
    likelihood = torch.softmax(logits / best_temperature, dim=-1)[:, 1]
    _, _, ece = calculate_ece(truth.to('cpu').numpy(),
                              preds.to('cpu').numpy(),
                              likelihood.to('cpu').numpy())
    print('ece: ', ece)


def main():
    experts = ['feature/feature',
               'content/bert_name', 'content/deberta_name',
               'content/bert_description', 'content/deberta_description',
               'content/bert_tweet', 'content/deberta_tweet',
               'ensemble/ensemble_bert', 'ensemble/ensemble_deberta']
    for expert in experts:
        data = torch.load(f'{expert}_validation.pt', weights_only=False)
        print(expert)
        calibration(data)


if __name__ == '__main__':
    main()
