import json
import os
import PIL
from transformers import ViTImageProcessor, Swinv2Model
from PIL import Image
import torch
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = ViTImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')
model = Swinv2Model.from_pretrained('microsoft/swinv2-tiny-patch4-window16-256')

model = model.to(device)


@torch.no_grad()
def obtain(path):
    image = Image.open(path)
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    out = model(**inputs)
    return out.pooler_output


def main():
    type_names = ['misinformation', 'verified_information', 'trend_information']
    image_dirs = [f'../raw_data/{item}/imgs' for item in type_names]
    for type_name, image_dir in zip(type_names, image_dirs):
        reps = []
        idx = []
        for item in tqdm(os.listdir(image_dir), desc=type_name, leave=False):
            try:
                rep = obtain(f'{image_dir}/{item}')
            except PIL.UnidentifiedImageError:
                continue
            reps.append(rep)
            idx.append(item)
        reps = torch.stack(reps).to('cpu')
        torch.save(reps, f'imgs_reps/{type_name}_reps.pt')
        json.dump(idx, open(f'imgs_reps/{type_name}_idx.json', 'w'))


if __name__ == '__main__':
    main()

