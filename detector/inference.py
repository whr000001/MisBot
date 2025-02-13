import torch
import json
import os
from dataset import NewsDataset, MySampler, get_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import NewsClassifier
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--run', type=int, default=5)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
lr = args.lr
fold = args.fold


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred) * 100, \
        precision_score(y_true, y_pred) * 100, recall_score(y_true, y_pred) * 100


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in tqdm(loader, leave=False):
        out, loss = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    metric = get_metric(all_truth, all_preds)
    for item in metric:
        print(f'{item:.2f}', end=' ')
    print()


def main():
    dataset = NewsDataset()
    for _ in range(10):
        fold_index = json.load(open(f'dataset/fold/{_}.json'))
        sampler = MySampler(fold_index, shuffle=True)
        loader = DataLoader(dataset, batch_size=9,
                            collate_fn=get_collate_fn(device), sampler=sampler)
        model = NewsClassifier(device=device, article=True, comment=True,
                               repost=True, image=True, video=False).to(device)
        checkpoint_dir = f'checkpoints/{_}'
        checkpoint_file = sorted(os.listdir(checkpoint_dir))[0]
        # print(checkpoint_file)
        checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}', weights_only=True)
        model.load_state_dict(checkpoint)
        inference(model, loader)


if __name__ == '__main__':
    main()
