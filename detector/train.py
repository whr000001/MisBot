import torch
import json
import os
from dataset import NewsDataset, MySampler, get_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import NewsClassifier
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run', type=int, default=5)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
lr = args.lr


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100


def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='train {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        out, loss = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        loss.backward()
        optimizer.step()
        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)
    ave_loss /= cnt
    print('train loss: {:.4f}'.format(ave_loss))
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, get_metric(all_truth, all_preds)


@torch.no_grad()
def validation(model, loader, epoch):
    model.eval()
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='valuate {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        out, _ = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return get_metric(all_truth, all_preds)


def train(train_loader, val_loader, test_loader, name):
    save_path = 'checkpoints/{}'.format(name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if len(os.listdir(save_path)) == 0:
        history_metrics = 0
    else:
        history_metrics = float(os.listdir(save_path)[0].replace('.pt', ''))

    model = NewsClassifier(device=device, article=True, comment=True,
                           repost=True, image=True, video=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up = 0
    for _ in range(100):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, _)
        acc = validation(model, val_loader, _)
        if acc > best_acc:
            best_acc = acc
            no_up = 0
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
        else:
            if train_acc >= 90:
                no_up += 1
        if no_up >= 4:
            break
        print('train loss: {:.2f}, train metric: {:.2f}, now best val metric: {:.2f}'.
              format(train_loss, train_acc, best_acc))
    model.load_state_dict(best_state)
    acc = validation(model, test_loader, 0)
    print('train {} done. val metric: {:.2f}, test metric: {:.2f}'.format(name, best_acc, acc))

    if acc > history_metrics:
        if os.path.exists(f'{save_path}/{history_metrics:.2f}.pt'):
            os.remove(f'{save_path}/{history_metrics:.2f}.pt')
        torch.save(best_state, f'{save_path}/{acc:.2f}.pt')


def main():
    dataset = NewsDataset()
    for fold in range(10):
        train_indices = []
        test_indices = []
        for _ in range(10):
            fold_index = json.load(open(f'dataset/fold/{_}.json'))
            if fold == _:
                test_indices += fold_index
            else:
                train_indices += fold_index
        train_sampler = MySampler(train_indices, shuffle=True)
        test_sampler = MySampler(test_indices, shuffle=False)
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  collate_fn=get_collate_fn(device), sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 collate_fn=get_collate_fn(device), sampler=test_sampler)
        train_name = f'{fold}'
        for _ in range(args.run):
            train(train_loader, test_loader, test_loader, train_name)


if __name__ == '__main__':
    main()
