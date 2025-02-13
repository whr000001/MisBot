import json
import numpy
from transformers import VideoMAEImageProcessor, VideoMAEModel, VideoMAEForVideoClassification
import torch
import av
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers.modeling_outputs import BaseModelOutput


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    frames = [x.to_ndarray(format="rgb24") for x in frames]
    resize_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        resized_pil_img = pil_img.resize((96, 96))
        resize_frame = np.array(resized_pil_img)
        resize_frames.append(resize_frame)

    return np.stack(resize_frames)


def sample_frame_indices(seg_len, k=256):
    indices = np.linspace(0, seg_len, k)
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    return indices


def obtain(path):
    container = av.open(path)
    indices = sample_frame_indices(seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    return video


def sample():
    type_names = ['misinformation', 'verified_information', 'trend_information']
    video_dirs = [f'../raw_data/{item}/video' for item in type_names]
    for type_name, video_dir in zip(type_names, video_dirs):
        save_dir = f'sampled_frames/{type_name}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for item in tqdm(os.listdir(video_dir), desc=type_name, leave=False):
            video_id = item.split('.')[0]
            save_path = f'{save_dir}/{video_id}.npy'
            if os.path.exists(save_path):
                continue
            try:
                rep = obtain(f'{video_dir}/{item}')
            except:
                continue
            numpy.save(save_path, rep)


def split(video):
    i = 0
    clips = []
    while i < len(video):
        clip = []
        st = i - 8
        ed = i + 8
        for j in range(st, ed):
            if 0 <= j < len(video):
                clip.append(video[j])
            else:
                clip.append(np.zeros((96, 96, 3)))
        # clip = np.stack(clip)
        clips.append(clip)
        i += 12
    return clips


def extract():
    image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base", local_files_only=True)
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", local_files_only=True).to(device)

    with torch.no_grad():
        type_names = ['misinformation', 'verified_information', 'trend_information']
        for type_name in type_names:
            data_dir = f'sampled_frames/{type_name}'
            files = os.listdir(data_dir)
            reps = []
            idx = []
            for file in tqdm(files, desc=type_name, leave=False):
                this_id = file.replace('.npy', '')
                video = np.load(f'{data_dir}/{file}')
                clips = split(video)
                inputs = image_processor(clips, return_tensors="pt", do_rescale=True, do_resize=False).to(device)
                outputs = model(**inputs)
                rep = outputs.last_hidden_state
                rep = rep[:, 0]
                rep = torch.mean(rep, dim=0)
                reps.append(rep)
                idx.append(this_id)
            reps = torch.stack(reps).to('cpu')
            torch.save(reps, f'video_reps/{type_name}_reps.pt')
            json.dump(idx, open(f'video_reps/{type_name}_idx.json', 'w'))


def main():
    sample()
    extract()


if __name__ == '__main__':
    main()
