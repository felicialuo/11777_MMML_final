import os
import pathlib
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments
import gc

import utils, data, videomae, transformerNet


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)


def train_one_epoch(train_loader, model, device, optimizer, criterion):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    for batch in train_loader:
        label = batch["labels"].to(device)
        total += label.shape[0]
        video_feats = videomae.get_videomae_feats(videomae_model, batch, device)

        logits, feat = model(video_feats)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

        # release memory
        del batch
        torch.cuda.empty_cache()
        gc.collect()

        losses.append(loss.item())

        print(f"Loss: {loss.item()}, Correct / total: {correct} / {total}")
    return losses, correct / total


def train(model, device, train_loader, optimizer, criterion, scheduler, num_epoch):
    model = model.to(device)
    losses = []
    train_accuracies = []

    for epoch in range(num_epoch):
        print("Epoch", epoch)
        loss, train_accuracy = train_one_epoch(train_loader, model, device, optimizer, criterion)
        losses.extend(loss)
        train_accuracies.append(train_accuracy)

        scheduler.step()

    



if __name__ == "__main__":
    import time
    start_time = time.time()


    # Example use
    dataset_root_path = '../datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)
    videomae_model = VideoMAEModel.from_pretrained(videomae_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset, test_seen_dataset, test_unseen_dataset = data.get_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)
    
    train_loader, test_seen_loader, test_unseen_loader = data.get_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, 
                                                                        image_processor, data.collate_fn, videomae_model, batch_size=16)
    
    model = transformerNet.TempNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(model, device, train_loader, optimizer, criterion, scheduler, num_epoch=5)




    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")



    
