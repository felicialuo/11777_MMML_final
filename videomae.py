import os
import pathlib
from tqdm import tqdm
import numpy as np

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments


import utils, data

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)




def get_videomae_feats(model, batch, device):
    inputs = {"pixel_values": batch["pixel_values"].to(device)}
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state # (B, 1568, 768)
    
    return feats



if __name__ == "__main__":
    import time
    start_time = time.time()


    # Example use
    dataset_root_path = '../datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)
    videomae_model = VideoMAEModel.from_pretrained(videomae_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_seen_dataset, test_unseen_dataset = data.get_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)
    
    train_loader, test_seen_loader, test_unseen_loader = data.get_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, 
                                                                        image_processor, data.collate_fn, videomae_model, batch_size=8)
    
    # print("\ntrain_loader")
    # i = 0
    # for batch in train_loader:
    #     feats = get_videomae_feats(videomae_model, batch, device)
    #     print(i, feats.shape)
    #     i += 1

    print("\ntest_seen_loader")
    i = 0
    for batch in test_seen_loader:
        feats = get_videomae_feats(videomae_model, batch, device)
        print(i, feats.shape)
        i += 1

    # print("\ntest_unseen_loader")
    # i = 0
    # for batch in test_unseen_loader:
    #     feats = get_videomae_feats(videomae_model, batch, device)
    #     print(i, feats.shape)
    #     i += 1




    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")



    
