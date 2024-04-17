import pathlib
import csv
import clip
import torch
from msclap import CLAP

from models import *

# def get_labels(root_path):
#     dataset_root_path = pathlib.Path(root_path)

#     video_count_train = len(list(dataset_root_path.glob('train/*/*.avi')))
#     video_count_test_seen = len(list(dataset_root_path.glob('test_seen/*/*.avi')))
#     video_count_test_unseen = len(list(dataset_root_path.glob('test_unseen/*/*.avi')))

#     all_video_file_paths = (
#         list(dataset_root_path.glob("train/*/*.avi"))
#         +
#         list(dataset_root_path.glob("test_seen/*/*.avi"))
#         +
#         list(dataset_root_path.glob("test_unseen/*/*.avi"))
#     )


#     class_labels = sorted({str(path).split("\\")[-2] for path in all_video_file_paths})
#     label2id = {label: i for i, label in enumerate(class_labels)}
#     id2label = {i: label for label, i in label2id.items()}

#     return label2id, id2label


def get_labels(csv_path="UCF101_AV_labels.csv"):
    label2id = {}
    id2label = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            id, label = row
            id = int(id)
            label2id[label] = id
            id2label[id] = label

    return label2id, id2label


def get_text_features(device, encoder_choice='CLIP'):
    # text_features = torch.load("UCF101_AV_text_features.pt").to(device)
    
    class_labels = get_labels()[0].keys()

    if encoder_choice == 'CLIP':
        clip_model, preprocess = clip.load("RN101", device)
        clip_model.eval()
        text_inputs = torch.cat([clip.tokenize(f"a video of {c}")for c in class_labels]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)

    elif encoder_choice == 'CLAP':
        clap_model = CLAP(version = '2023', use_cuda=(device == "cuda"))
        with torch.no_grad():
            text_features = clap_model.get_text_embeddings([f"a video of {c}"for c in class_labels])

    else: 
        print("Unsupported encoder choice, choose from 'CLIP', 'CLAP'")
        return

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.float()

def load_audio_resnet(layers: int, resnet_ckpt: str | None, freeze: bool, num_classes: int) -> torch.nn.Module:
    net = AudioResNet(layers, freeze, num_classes)
    if resnet_ckpt is not None:
        net.load(resnet_ckpt)
    
    return net