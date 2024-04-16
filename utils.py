import pathlib
import csv
import clip
import torch

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

def get_text_features(device):
    # text_features = torch.load("UCF101_AV_text_features.pt").to(device)
    clip_model, preprocess = clip.load("RN101", device)
    clip_model.eval()
    class_labels = get_labels()[0].keys()
    text_inputs = torch.cat([clip.tokenize(f"a video of {c}")for c in class_labels]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)

    return text_features.float()