import os
import pathlib
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import gc

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments
import clip


import utils, data, videomae, transformerNet

import argparse

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)


def train_one_epoch(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    batch_idx = 0
    for batch in tqdm(train_loader, total=train_loader.dataset.num_videos//train_loader.batch_size+1, position=0, leave=False, desc="Train Epoch {}".format(epoch)):
        label = batch["labels"].to(device)
        total += label.shape[0]

        logits_per_av, logits_per_text, av_features, text_features = model(batch)
        loss = criterion(logits_per_av, label)
        loss.backward()
        optimizer.step()
        
        pred = logits_per_av.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

        # release memory
        del batch
        torch.cuda.empty_cache()
        gc.collect()

        losses.append(loss.item())

        if batch_idx % 50 == 0:
            tqdm.write('\tTrain Set Batch {}: Current loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                        batch_idx, loss.item(), correct, total,
                        100. * correct / total))
        batch_idx += 1

    average_loss = np.mean(losses)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        average_loss, correct, total,
        100. * correct / total))
    return average_loss, correct / total


def eval(model, test_loader, device, criterion, epoch):
    model.eval()
    losses = []
    correct = 0
    total = 0

    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, total=test_loader.dataset.num_videos//test_loader.batch_size+1, position=0, leave=False, desc="Test Epoch {}".format(epoch)):
            label = batch["labels"].to(device)
            total += label.shape[0]

            logits_per_av, logits_per_text, av_features, text_features = model(batch)

            loss = criterion(logits_per_av, label)
            pred = logits_per_av.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
            losses.append(loss.item())

            if batch_idx % 10 == 0:
                tqdm.write('\tTest Set Batch {}: Current loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                            batch_idx, loss.item(), correct, total,
                            100. * correct / total))
            batch_idx += 1

    # total = test_loader.dataset.num_videos
    average_loss = np.mean(losses)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        average_loss, correct, total,
        100. * correct / total))
    return average_loss, correct / total


def train(model, device, train_loader, test_seen_loader, test_unseen_loader, optimizer, criterion, scheduler, num_epoch):
    model = model.to(device)
    train_losses = [] # all one value per epoch
    train_accuracies = []
    test_seen_losses = []
    test_seen_accuracies = []
    test_unseen_losses = []
    test_unseen_accuracies = []

    for epoch in range(num_epoch):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch)
        train_losses.append(train_loss) # average loss of every epoch
        train_accuracies.append(train_accuracy)

        test_seen_loss, test_seen_accuracy = eval(model, test_seen_loader, device, criterion, epoch)
        test_seen_losses.append(test_seen_loss) # average loss of every epoch
        test_seen_accuracies.append(test_seen_accuracy)

        test_unseen_loss, test_unseen_accuracy = eval(model, test_unseen_loader, device, criterion, epoch)
        test_unseen_losses.append(test_unseen_loss) # average loss of every epoch
        test_unseen_accuracies.append(test_unseen_accuracy)

        scheduler.step()

    return train_losses, train_accuracies, test_seen_losses, test_seen_accuracies, test_unseen_losses, test_unseen_accuracies

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Machine Learning Project, Human Activity Recognition')
    
    parser.add_argument("--data_path", type=str, default="./datasets/UCF101/UCF-101/")
    parser.add_argument("--videomae_ckpt", type=str, default="MCG-NJU/videomae-base")

    parser.add_argument("--audiores_ckpt", type=str, default="./Models/AudioResNet/finetuned-L18.pth")
    parser.add_argument("--audiores_layers", type=int, default=18,
                        choices=[18, 34, 50, 101, 134])

    parser.add_argument("--text_encoder", type=str, default="CLIP",
                        choices=["CLIP", "CLAP"])
    
    parser.add_argument("--network", type=str, default="TempNet",
                        choices=["TempNet", "VCLAPNet"])
    parser.add_argument("--num_epoch", type=int, default=5)

    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    import time
    start_time = time.time()

    # Example Use
    image_processor = VideoMAEImageProcessor.from_pretrained(args.videomae_ckpt)
    videomae_model = VideoMAEModel.from_pretrained(args.videomae_ckpt)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset, test_seen_dataset, test_unseen_dataset = data.get_iterable_dataset(args.data_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)
    
    train_loader, test_seen_loader, test_unseen_loader = data.get_iterable_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, 
                                                                        image_processor, data.collate_fn, videomae_model, batch_size=8)
    
    # train_dataset, test_seen_dataset, test_unseen_dataset = data.get_default_dataset(dataset_root_path)
    # train_loader, test_seen_loader, test_unseen_loader = data.get_default_dataloader(train_dataset, test_seen_dataset, \
    #                                                                                  test_unseen_dataset, data.collate_fn, batch_size=16)
    
    
    # text features
    text_features = utils.get_text_features(device, encoder_choice=args.text_encoder)

    if args.network == "TempNet": model = transformerNet.TempNet(videomae_model, text_features, av_emb_size=768, device=device)
    elif args.network == "VCLAPNet": model = transformerNet.VCLAPNet(text_features, av_emb_size=512, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, train_accuracies, \
        test_seen_losses, test_seen_accuracies, \
            test_unseen_losses, test_unseen_accuracies = train(model, device, train_loader, test_seen_loader, test_unseen_loader, 
                                                                optimizer, criterion, scheduler, num_epoch=args.num_epoch)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")



    
