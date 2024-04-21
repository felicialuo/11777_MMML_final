import pathlib
import csv
import clip
import torch
from msclap import CLAP
import numpy as np
import random

from torchaudio.compliance import kaldi
import torch.nn.functional as F

import os
import sys

import pathlib
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

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


def get_labels(csv_path="csv/UCF101_AV_labels.csv"):
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

def freeze(model: torch.nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    
    return model

def named_freeze(model: torch.nn.Module, names: list, full_match: bool = True) -> torch.nn.Module:
    for name, p in model.named_parameters():
        if full_match and name in names:
            p.requires_grad = False
        else:
            for sub in names:
                if sub in name:
                    p.requires_grad = False
                break
    
    return model

def get_videomae_feats(model, batch, device, freeze=True):
    inputs = {"pixel_values": batch["pixel_values"].to(device)}
    model = model.to(device)
    
    if freeze:
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state # (B, 1568, 768)
    else:
        outputs = model(**inputs)
        feats = outputs.last_hidden_state # (B, 1568, 768)

    
    return feats

def load_audio_into_tensor(audio_time_series: torch.Tensor, sample_rate: int, audio_duration: int, mae_feature: bool) -> torch.Tensor:
    r"""Loads raw audio tensors."""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    # audio_time_series = audio_time_series.reshape(-1)
    if audio_time_series.shape[0] == 2:
        audio_time_series = audio_time_series.mean(dim=0, keepdim=True)
    audio_length = audio_time_series.size(1)

    # audio_time_series is shorter than predefined audio duration,
    # so audio_time_series is extended
    if audio_duration * sample_rate >= audio_length:
        repeat_factor = int(np.ceil((audio_duration * sample_rate) / audio_length))
        # Repeat audio_time_series by repeat_factor to match audio_duration
        audio_time_series = audio_time_series.repeat(1, repeat_factor)
        # remove excess part of audio_time_series
        audio_time_series = audio_time_series[:, 0:audio_duration*sample_rate]
    else:
        # audio_time_series is longer than predefined audio duration,
        # so audio_time_series is trimmed
        start_index = random.randrange(audio_length - audio_duration*sample_rate)
        audio_time_series = audio_time_series[:, start_index : start_index + audio_duration * sample_rate]

    if mae_feature:
        # need to convert to FBank
        # hard code the parameters
        target_len, num_mel_bins = 1024, 128
        spectro = kaldi.fbank(audio_time_series, htk_compat=True, use_energy=False, 
                window_type='hanning', num_mel_bins=num_mel_bins, 
                dither=0.0, frame_shift=3.9, frame_length=30, sample_frequency=sample_rate)

        if spectro.size(0) < 1024:
            num_pad = target_len - spectro.shape[0]
            # spectro = torch.concatenate((spectro, torch.zeros((num_pad, spectro.shape[1]))))
            spectro = F.pad(spectro, (0, 0, 0, num_pad), mode="constant", value=0)
        else:
            spectro = spectro[:target_len]
        audio_tensor = spectro
    else:
        audio_tensor = audio_time_series.squeeze()
    
    return audio_tensor

def load_batch_audio_into_tensor(audio_batch: torch.Tensor, sample_rate: int, audio_duration: int, mae_feature: bool) -> torch.Tensor:
    r"""Loads a batch of audio and process them in length."""
    processed_batch = []
    for audio_time_series in audio_batch:
        processed_audio = load_audio_into_tensor(audio_time_series, sample_rate, audio_duration, mae_feature)
        processed_batch.append(processed_audio)
    return torch.stack(processed_batch).unsqueeze(1)

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    sys.path.append("AudioMAE")
    import models_mae
    # build model
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True,img_size=(1024,128),decoder_mode=1,decoder_depth=16)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    return model

if __name__ == "__main__":
    seen_label2id, seen_id2label, unseen_label2id, unseen_id2label = get_sep_seen_unseen_labels()
