import pathlib
import csv
import clip
import torch
#from msclap import CLAP

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

def get_sep_seen_unseen_labels(dataset_root_path = './datasets/UCF101/UCF-101/'):

    dataset_root_path = pathlib.Path(dataset_root_path)
    unseen_labels = sorted({str(path).split("\\")[-2] for path in list(dataset_root_path.glob("test_unseen/*/*.avi"))})

    label2id, id2label = get_labels()
    seen_label2id, seen_id2label = {label: id for label, id in label2id.items() if label not in unseen_labels},\
                             {id: label for label, id in label2id.items() if label not in unseen_labels}
    unseen_label2id, unseen_id2label = {label: id for label, id in label2id.items() if label in unseen_labels},\
                                {id: label for label, id in label2id.items() if label in unseen_labels}
    
    return seen_label2id, seen_id2label, unseen_label2id, unseen_id2label


def get_id2id_mapping(dataset_root_path = './datasets/UCF101/UCF-101/'):

    label2id, id2label = get_labels()

    dataset_root_path = pathlib.Path(dataset_root_path)

    seen_labels = sorted({str(path).split("\\")[-2] for path in list(dataset_root_path.glob("test_seen/*/*.avi"))})
    seen_label2id = {label: i for i, label in enumerate(seen_labels)}
    seen_id2label = {i: label for label, i in seen_label2id.items()}


    unseen_labels = sorted({str(path).split("\\")[-2] for path in list(dataset_root_path.glob("test_unseen/*/*.avi"))})
    unseen_label2id = {label: i for i, label in enumerate(unseen_labels)}
    unseen_id2label = {i: label for label, i in unseen_label2id.items()}

    seen_id2all_id = {i: label2id[label] for i, label in seen_id2label.items()}
    unseen_id2all_id = {i: label2id[label] for i, label in unseen_id2label.items()}

    return seen_id2all_id, unseen_id2all_id

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
        clap_model = CLAP(version = '2023', use_cuda=True)
        with torch.no_grad():
            text_features = clap_model.get_text_embeddings([f"a video of {c}"for c in class_labels])

    else: 
        print("Unsupported encoder choice, choose from 'CLIP', 'CLAP'")
        return

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.float()

if __name__ == "__main__":
    seen_label2id, seen_id2label, unseen_label2id, unseen_id2label = get_sep_seen_unseen_labels()
