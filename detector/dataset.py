import json
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data


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


class NewsDataset(Dataset):
    def __init__(self):
        data = json.load(open('dataset/main_data.json'))
        main_data = torch.load('dataset/main_data.pt', weights_only=True, map_location='cpu')
        video = torch.load('dataset/videos_data.pt', weights_only=True, map_location='cpu')
        image = torch.load('dataset/images_data.pt', weights_only=True, map_location='cpu')
        self.data = []
        for index, item in enumerate(data):
            self.data.append({
                'content': main_data[index]['content'].squeeze(0),
                'repost_x': main_data[index]['repost_x'],
                'repost_edge': main_data[index]['repost_edge'],
                'comment_x': main_data[index]['comment_x'],
                'comment_edge': main_data[index]['comment_edge'],
                'label': item['label'],
                'video': video[index].squeeze(0),
                'image': image[index].squeeze(0)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_collate_fn(device):
    def collate_fn(batch):
        content = []

        repost = []
        comment = []
        comment_graph_batch = []

        image = []
        video = []

        label = []
        for index, item in enumerate(batch):
            content.append(item['content'].to(device))
            repost.append(Data(x=torch.cat([torch.zeros(1, 768, dtype=torch.float), item['repost_x']]),
                               edge_index=item['repost_edge'].to(torch.long)).to(device))

            comment_edge = item['comment_edge']
            if isinstance(comment_edge, list):
                graph_size = []
                for graph in comment_edge:
                    graph_size.append(graph.shape[1] + 1)
                comment_x = torch.split(item['comment_x'], graph_size)
                for x, edge in zip(comment_x, comment_edge):
                    comment.append(Data(x=x, edge_index=edge.to(torch.long)).to(device))
                    comment_graph_batch.append(index)
            image.append(item['image'].to(device))
            video.append(item['video'].to(device))
            label.append(item['label'])

        comment_graph_batch = torch.tensor(comment_graph_batch, dtype=torch.long).to(device)
        image = torch.stack(image)
        video = torch.stack(video)
        content = torch.stack(content)
        label = torch.tensor(label, dtype=torch.long).to(device)
        return {
            'content': content,
            'repost': repost,
            'comment': comment,
            'image': image,
            'video': video,
            'label': label,
            'comment_graph_batch': comment_graph_batch,
            'batch_size': len(batch)
        }
    return collate_fn
