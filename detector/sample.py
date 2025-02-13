import os
import json
import torch
import random
from tqdm import tqdm
from torch_geometric.utils import subgraph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(20250101)


def importance(data):
    if isinstance(data, str):
        return len(set(data))
    imp_0 = len(data[0])
    imp_1 = 0
    for item in data[0]:
        imp_1 += len(set(item))
    return imp_0, imp_1


def sample_graph(node, edge, k=20):
    if len(edge) == 0:
        return [[None], [[], []]]
    node_with_importance = [(index + 1, item, importance(item)) for index, item in enumerate(node)]
    sorted_node_with_importance = sorted(node_with_importance, key=lambda x: x[-1], reverse=True)
    father = {}
    for src, tgt in edge:
        father[src] = tgt
    subset = set()
    for node in sorted_node_with_importance:
        node = node[0]
        rt = node
        rt_list = [rt]
        while True:
            rt = father[rt]
            if not rt:
                break
            rt_list.append(rt)
        rt_list = list(reversed(rt_list))
        for item in rt_list:
            if len(subset) < k:
                subset.add(item)
    subset.add(0)
    subset = list(subset)
    subset = torch.tensor(subset, dtype=torch.long).to(device)
    edge_index = torch.tensor(edge, dtype=torch.long).to(device).transpose(1, 0)
    edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
    content = [None]
    for item in subset[1:]:
        content.append(node_with_importance[item-1][1])
    edge_index = edge_index.to('cpu').numpy().tolist()
    return [content, edge_index]


def main():
    clusters = json.load(open('../topic/data/clusters.json'))
    misinformation_idx = json.load(open('../topic/data/misinformation_idx.json'))
    print(len(clusters), len(misinformation_idx))
    folds = [[] for _ in range(10)]
    fold_sums = [0] * 10

    for cluster in clusters:
        min_index = fold_sums.index(min(fold_sums))
        folds[min_index] += cluster
        fold_sums[min_index] += len(cluster)

    misinformation_id2fold_id = {}

    for fold_index, fold in enumerate(folds):
        for item in fold:
            misinformation_id2fold_id[misinformation_idx[item]] = fold_index

    main_data = []
    video_data = []
    image_data = []
    belong_to = []

    for _ in ['misinformation', 'verified_information', 'trend_information']:
        videos = torch.load(f'video_reps/{_}_reps.pt', weights_only=True)
        video_idx = json.load(open(f'video_reps/{_}_idx.json'))
        assert len(videos) == len(video_idx)
        id2video = {}
        for vid, rep in zip(video_idx, videos):
            vid = vid.split('_')[0]
            if vid not in id2video:
                id2video[vid] = []
            id2video[vid].append(rep)

        images = torch.load(f'imgs_reps/{_}_reps.pt', weights_only=True)
        image_idx = json.load(open(f'imgs_reps/{_}_idx.json'))
        assert len(images) == len(image_idx)
        id2image = {}
        for iid, rep in zip(image_idx, images):
            iid = iid.split('_')[0]
            if iid not in id2image:
                id2image[iid] = []
            id2image[iid].append(rep)

        data_dir = f'../data/{_}'
        files = os.listdir(data_dir)
        if _ != 'misinformation':
            files = sorted(files)
            random.shuffle(files)

        for file_index, file in enumerate(tqdm(files, desc=f'{_}', leave=False)):
            nid = file.replace('.json', '')
            if _ == 'misinformation' and nid not in misinformation_id2fold_id:
                continue
            data = json.load(open(f'{data_dir}/{file}'))
            repost_graph = data['repost_graph']
            repost_content = []
            for node in repost_graph['nodes']:
                repost_content.append(node['text'])
            repost_edge = repost_graph['edges']
            repost_graph = sample_graph(repost_content, repost_edge, k=50)
            repost_graph[0][0] = data['article']['article_content']

            comments = []
            for graph in data['comment_graphs']:
                comment_content = []
                comment_edge = graph['edges']
                for node in graph['nodes']:
                    comment_content.append(node['text'])
                comments.append([comment_content, comment_edge])
            comments_with_importance = [(item, importance(item)) for item in comments]
            comments = sorted(comments_with_importance, key=lambda x: x[-1], reverse=True)
            comments = [item[0] for item in comments]
            comment_graphs = []
            total_size = 0
            for item in comments:
                graph = sample_graph(item[0][1:], item[1], k=20)
                graph[0][0] = item[0][0]
                comment_graphs.append(graph)
                total_size += len(graph[0])
                if total_size >= 200:
                    break
            main_data.append({
                'content': data['article']['article_content'],
                'repost_graph': repost_graph,
                'comment_graphs': comment_graphs,
                'label': int(_ == 'misinformation'),
            })
            belong_to.append(
                file_index % 10 if _ != 'misinformation' else misinformation_id2fold_id[nid]
            )
            if nid in id2video:
                video = id2video[nid]
                video = torch.stack(video).mean(0)
            else:
                video = torch.zeros(768, dtype=torch.float)
            video_data.append(video)

            if nid in id2image:
                image = id2image[nid]
                image = torch.stack(image).mean(0)
            else:
                image = torch.zeros(768, dtype=torch.float)
            image_data.append(image)
    print(len(main_data), len(belong_to), len(video_data), len(image_data))
    torch.save(image_data, 'dataset/images_data.pt')
    torch.save(video_data, 'dataset/videos_data.pt')
    json.dump(main_data, open('dataset/main_data.json', 'w'))
    fold_index = [[] for _ in range(10)]
    for index, belong in enumerate(belong_to):
        fold_index[belong].append(index)
    for item in fold_index:
        print(len(item), end=',')
    for _, fold in enumerate(fold_index):
        json.dump(fold, open(f'dataset/fold/{_}.json', 'w'))


if __name__ == '__main__':
    main()
