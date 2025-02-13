import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_scatter import scatter


class GCNEncoder(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.gnn1 = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim, bias=False)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gnn1(x, edge_index)
        x = self.act_fn(self.dropout(x))
        x = self.gnn2(x, edge_index)
        return x


class NewsClassifier(nn.Module):
    def __init__(self, device, hidden_dim=256, dropout=0.5,
                 article=True, comment=True, repost=True, image=True, video=True):
        super().__init__()
        self.text_encoder = nn.Linear(768, hidden_dim, bias=False)
        self.video_encoder = nn.Linear(768, hidden_dim, bias=False)
        self.image_encoder = nn.Linear(768, hidden_dim, bias=False)
        self.repost_graph_encoder = GCNEncoder()
        self.comment_graph_encoder = GCNEncoder()

        self.article = article
        self.comment = comment
        self.repost = repost
        self.image = image
        self.video = video

        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        content_reps = self.text_encoder(data['content'])

        if self.repost:
            repost = Batch.from_data_list(data['repost'])
            repost_x = self.text_encoder(repost['x'])
            repost_reps = self.repost_graph_encoder(repost_x, repost['edge_index'])
            repost_reps = global_mean_pool(repost_reps, repost['batch'])
        else:
            repost_reps = torch.zeros(data['batch_size'], self.hidden_dim, dtype=torch.float).to(self.device)

        if self.comment:
            comment = data['comment']
            if len(comment) == 0:
                comment_reps = torch.zeros(data['batch_size'], self.hidden_dim, dtype=torch.float).to(self.device)
            else:
                comment = Batch.from_data_list(comment)
                comment_x = self.text_encoder(comment['x'])
                graphs_reps = self.comment_graph_encoder(comment_x, comment['edge_index'])
                graphs_reps = global_mean_pool(graphs_reps, comment['batch'])
                comment_reps = scatter(graphs_reps, data['comment_graph_batch'],
                                       dim=-2, dim_size=data['batch_size'], reduce='mean')
        else:
            comment_reps = torch.zeros(data['batch_size'], self.hidden_dim, dtype=torch.float).to(self.device)

        if self.video:
            video_reps = self.dropout(self.video_encoder(data['video']))
        else:
            video_reps = torch.zeros(data['batch_size'], self.hidden_dim, dtype=torch.float).to(self.device)

        if self.image:
            image_reps = self.dropout(self.image_encoder(data['image']))
        else:
            image_reps = torch.zeros(data['batch_size'], self.hidden_dim, dtype=torch.float).to(self.device)

        reps = torch.cat([content_reps, repost_reps, comment_reps, video_reps, image_reps], dim=-1)
        preds = self.cls(reps)
        loss = self.loss_fn(preds, data['label'])
        return preds, loss
