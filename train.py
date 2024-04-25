import os
import pathlib
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments
import clip
from msclap import CLAP
from models import AlignNet, TempNet, VCLAPNet, AudioMAEWrapper

import utils, data, loss_utils

import argparse

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

# import os
# import sys
# sys.path.append("AudioMAE")
# import models_mae

def train_one_epoch(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    model.enable_lora()
    losses = []
    correct = 0
    total = 0
    
    batch_idx = 0
    for batch in tqdm(train_loader, total=train_loader.dataset.num_videos//train_loader.batch_size+1, position=0, leave=False, desc="Train Epoch {}".format(epoch)):
        label = batch["labels"].to(device)
        total += label.shape[0]

        logits_per_av, logits_per_text, av_features, text_features = model(batch)

        if criterion == 'ce':
            loss = F.cross_entropy(logits_per_av, label)
        elif criterion == 'symmetric_ce':
            loss = loss_utils.symmetric_ce(logits_per_av, logits_per_text, label)
        elif criterion == 'composite_loss':
            loss = loss_utils.composite_loss(logits_per_av, logits_per_text, av_features, text_features, label)

        loss.backward()
        optimizer.step()
        
        pred = logits_per_av.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

        # # release memory
        # del batch
        # torch.cuda.empty_cache()
        # gc.collect()

        losses.append(loss.item())

        if batch_idx % 10 == 0:
            tqdm.write('\tTrain Set Batch {}: Current loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                        batch_idx, loss.item(), correct, total,
                        100. * correct / total))
        batch_idx += 1

    average_loss = np.mean(losses)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        average_loss, correct, total,
        100. * correct / total))
    return average_loss, correct / total


def eval(model, test_loader, device, criterion, epoch, seen=True):
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    if seen:
        model.enable_lora()
    else:
        model.disable_lora()

    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, total=test_loader.dataset.num_videos//test_loader.batch_size+1, position=0, leave=False, desc="Test Epoch {}".format(epoch)):
            label = batch["labels"].to(device)
            total += label.shape[0]

            logits_per_av, logits_per_text, av_features, text_features = model(batch)

            if criterion == 'ce':
                loss = nn.CrossEntropyLoss()(logits_per_av, label)
            elif criterion == 'symmetric_ce':
                loss = loss_utils.symmetric_ce(logits_per_av, logits_per_text, label)
            elif criterion == 'composite_loss':
                loss = loss_utils.composite_loss(logits_per_av, logits_per_text, av_features, text_features, label)

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

def save_model(model, optimizer, scheduler, epoch, ckpt_name):
    if not os.path.exists(ckpt_name):
        os.makedirs(ckpt_name)
    torch.save(
        {'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch},
        os.path.join(ckpt_name, f'epoch{epoch}.pt'))


def train(model, device, train_loader, test_seen_loader, test_unseen_loader, optimizer, criterion, scheduler, num_epoch, ckpt_name):
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

        test_unseen_loss, test_unseen_accuracy = eval(model, test_unseen_loader, device, criterion, epoch, seen=False)
        test_unseen_losses.append(test_unseen_loss) # average loss of every epoch
        test_unseen_accuracies.append(test_unseen_accuracy)

        save_model(model, optimizer, scheduler, epoch, ckpt_name)

        scheduler.step()

    return train_losses, train_accuracies, test_seen_losses, test_seen_accuracies, test_unseen_losses, test_unseen_accuracies

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Machine Learning Project, Human Activity Recognition')
    
    parser.add_argument("--dataset", type=str, default="ucf",
                        choices=["ucf", "kinetics"])
    parser.add_argument("--data_path", type=str, default="./datasets/UCF101/UCF-101/")
    parser.add_argument("--videomae_ckpt", type=str, default="MCG-NJU/videomae-base")

    parser.add_argument("--audiores_ckpt", type=str, default="./Models/AudioResNet/finetuned-L18.pth")
    parser.add_argument("--audiores_layers", type=int, default=18,
                        choices=[18, 34, 50, 101, 134])
    parser.add_argument("--audiomae_arch", type=str, default="mae_vit_base_patch16")
    parser.add_argument("--audiomae_ckpt", type=str, default="D:/Models/AudioMAE/pretrained.pth")

    parser.add_argument("--save_ckpt_to", type=str, default="ckpts/")

    parser.add_argument("--text_encoder", type=str, default="CLIP",
                        choices=["CLIP", "CLAP"])
    parser.add_argument("--clip_arch", type=str, default="ViT-L/14",
                        choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    
    parser.add_argument("--network", type=str, default="TempNet",
                        choices=["TempNet", "VCLAPNet", "AlignNet"])
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-6)

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_iterable_DS", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_videomae", action="store_true")
    parser.add_argument("--use_audiomae", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_prompt_learner", action="store_true")
    parser.add_argument("--use_temporal_audio", action="store_true")


    parser.add_argument("--loss", type=str, default='ce',
                        choices=['ce', 'symmetric_ce', 'composite_loss'])

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    import time
    start_time = time.time()

    # Example Use
    image_processor = VideoMAEImageProcessor.from_pretrained(args.videomae_ckpt)
    videomae_model = VideoMAEModel.from_pretrained(args.videomae_ckpt)
    utils.freeze(videomae_model)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    if args.use_iterable_DS:
        train_dataset, test_seen_dataset, test_unseen_dataset = data.get_iterable_dataset(args.dataset, args.data_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)        
        train_loader, test_seen_loader, test_unseen_loader = data.get_iterable_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, 
                                                                            image_processor, data.collate_fn, videomae_model, batch_size=args.batch_size)
    else:
        train_dataset, test_seen_dataset, test_unseen_dataset = data.get_default_dataset(args.data_path)
        train_loader, test_seen_loader, test_unseen_loader = data.get_default_dataloader(train_dataset, test_seen_dataset, \
                                                                                        test_unseen_dataset, data.collate_fn, batch_size=16)     

    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)                                                                                    
    
    
    # text features
    text_features = utils.get_text_features(device, encoder_choice=args.text_encoder)
    if args.dataset == 'ucf':
        classname = list(utils.get_labels()[0].keys())
    elif args.dataset == 'kinetics':
        classname = list(utils.get_labels(csv_path="csv/Kinetics400_labels.csv")[0].keys())

    # clip model
    clip_model, _ = clip.load(args.clip_arch, device)
    clip_model.float()
    #clip_model.eval()
    #utils.freeze(clip_model)
    

    # clap model
    clap_model, audiomae_model = None, None
    if args.use_audio:
        if args.use_audiomae:
            audiomae_model = AudioMAEWrapper(args.audiomae_ckpt, args.audiomae_arch, False).to(device)
        else:
            clap_model = CLAP(version = '2023', use_cuda=args.use_gpu)

    if args.network == "TempNet": 
        model = TempNet(videomae_model, text_features, av_emb_size=768, device=device)
    elif args.network == "VCLAPNet": 
        model = VCLAPNet(videomae_model, audiomae_model, classname, clip_model, clap_model, device, use_videomae=args.use_videomae, use_audio=args.use_audio,
                         use_audiomae=args.use_audiomae, use_temporal_audio=args.use_temporal_audio)
        model.freeze(visual=True, audio=True, text=True)
    elif args.network == "AlignNet": 
        model = AlignNet(videomae_model, classname, clip_model, device, use_videomae=args.use_videomae)
        model.freeze(visual=True, audio=True, text=True)
        
    del image_processor
    del videomae_model
    torch.cuda.empty_cache()
    gc.collect()
    
    if args.use_lora:
        model.add_lora()
        

    print(model)
    for name, param in model.image_encoder.named_parameters():
            print(name, param.requires_grad)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = args.loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    train_losses, train_accuracies, \
        test_seen_losses, test_seen_accuracies, \
            test_unseen_losses, test_unseen_accuracies = train(model, device, train_loader, test_seen_loader, test_unseen_loader, 
                                                                optimizer, criterion, scheduler, num_epoch=args.num_epoch, 
                                                                ckpt_name=args.save_ckpt_to)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")