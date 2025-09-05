import glob
import json


def load_user():
    with open('train_data.jsonl') as f:  # pls enter your own file oath
        for line in f.readlines():
            data = json.loads(line)
            print(data.keys())  # tweet, description, numerical, categorical, label, uid
            print(data['tweet'])  # the list of published posts
            print(data['description'])  # the description of this user
            print(data['numerical'])  # followers count; follow count; and statuses count 3 dims
            print(data['categorical'])  # onehot encoding 20 dims
            # [0 1] if not verified else [1 0] 2 dims
            # [0 1] if not svip else [1 0] 2 dims
            # mbrank ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 10 dims
            # mb_type_name ['0', '2', '11', '12', '13', '14'] 6 dims
            break
    with open('inference_data.jsonl') as f:  # pls enter your own file oath
        for line in f.readlines():
            data = json.loads(line)
            print(data.keys())  # tweet, description, numerical, categorical, uid
            break
    labels = json.load(open('inference_labels.json'))
    for uid, label in labels.items():
        print(uid, label)
        # label is a list with 2 items, the first is the prediction (1 for bot and 0 for human),
        # the second is the bot score (we consider a user a bot if the score >= 0.75)
        break


def load_information():
    for sample_name in ['misinformation', 'verified_information', 'trend_information']:
        cnt = 0
        with open(f'{sample_name}.jsonl') as f:  # pls enter your own file oath
            for line in f.readlines():
                instance_id = f'{sample_name}_{cnt:04}'
                data = json.loads(line)
                print(data.keys())  # article, comment_graphs, repost_graph, comment_users, repost_users, attitude_users
                print(data['comment_users'])
                print(data['repost_users'])
                print(data['attitude_users'])
                article = data['article']
                print(article.keys())  # article_content, repost_count, comment_count, attitude_count, publish_time

                #  load related images
                image_path = f'imgs/{sample_name}/{instance_id}*'  # pls enter your own file oath
                files = glob.glob(image_path)
                print(files)

                #  load related videos
                image_path = f'videos/{sample_name}/{instance_id}*'  # pls enter your own file oath
                files = glob.glob(image_path)
                print(files)
                cnt += 1

                break


def main():
    load_user()
    load_information()


if __name__ == '__main__':
    main()
