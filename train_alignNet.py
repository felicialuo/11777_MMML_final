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


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

clip_templates = {
    'ApplyEyeMakeup': 'A person applying eye makeup carefully with brushes and cosmetics.',
    'ApplyLipstick': 'An individual applying lipstick, focusing on the details of the lips.',
    'Archery': 'A person holding a bow and aiming an arrow in an archery stance.',
    'BabyCrawling': 'A baby crawling on the floor, moving forward on hands and knees.',
    'BalanceBeam': 'An athlete performing on a balance beam, displaying agility and balance.',
    'BandMarching': 'A marching band moving in formation while playing instruments.',
    'BasketballDunk': 'A basketball player performing a dunk, leaping high towards the basket.',
    'BlowDryHair': 'Someone blow drying hair, using a hairdryer to style it.',
    'BlowingCandles': 'A person blowing out candles on a cake, possibly making a wish.',
    'BodyWeightSquats': 'An individual doing body weight squats, showing exercise and fitness.',
    'Bowling': 'A bowler throwing a bowling ball down a lane towards pins.',
    'BoxingPunchingBag': 'A boxer practicing punches on a punching bag.',
    'BoxingSpeedBag': 'A boxer training with a speed bag, improving hand-eye coordination.',
    'BrushingTeeth': 'A person brushing their teeth with a toothbrush and toothpaste.',
    'CliffDiving': 'An adventurer cliff diving into water from a high cliff.',
    'CricketBowling': 'A cricket bowler in action, delivering the ball to the batsman.',
    'CricketShot': 'A cricket batsman playing a shot with the cricket bat.',
    'CuttingInKitchen': 'Someone cutting vegetables or ingredients in a kitchen setting.',
    'FieldHockeyPenalty': 'A field hockey player taking a penalty shot at goal.',
    'FloorGymnastics': 'A gymnast performing floor gymnastics, showing flexibility and strength.',
    'FrisbeeCatch': 'A person catching a frisbee, showcasing agility and timing.',
    'FrontCrawl': 'A swimmer doing the front crawl stroke in a pool.',
    'Haircut': 'A professional giving a haircut to a client, using scissors and combs.',
    'HammerThrow': 'An athlete throwing a hammer in a hammer throw competition.',
    'Hammering': 'A person hammering a nail into wood or a similar surface.',
    'HandStandPushups': 'An individual doing handstand pushups, showing strength and balance.',
    'HandstandWalking': 'Someone walking on their hands, maintaining a handstand position.',
    'HeadMassage': 'A person receiving a relaxing head massage.',
    'IceDancing': 'Ice dancers performing a routine on an ice rink with grace and skill.',
    'Knitting': 'An individual knitting with yarn and needles, creating a textile.',
    'LongJump': 'An athlete performing a long jump, leaping into the sand pit.',
    'MoppingFloor': 'Someone mopping a floor, cleaning it with a mop and bucket.',
    'ParallelBars': 'A gymnast performing a routine on the parallel bars.',
    'PlayingCello': 'A musician playing the cello, using a bow on the strings.',
    'PlayingDaf': 'A person playing the daf, a type of frame drum.',
    'PlayingDhol': 'An individual playing the dhol, a traditional drum.',
    'PlayingFlute': 'A flutist playing the flute, producing melodious music.',
    'PlayingSitar': 'A musician playing the sitar, an Indian stringed instrument.',
    'Rafting': 'Adventurers rafting down a river, navigating rapids and waves.',
    'ShavingBeard': 'A person shaving their beard with a razor or electric shaver.',
    'Shotput': 'An athlete throwing a shot put in a track and field event.',
    'SkyDiving': 'A skydiver free falling or parachuting from high altitude.',
    'SoccerPenalty': 'A soccer player taking a penalty kick at goal.',
    'StillRings': 'A gymnast performing on the still rings, showing strength and control.',
    'SumoWrestling': 'Sumo wrestlers engaging in a match, showcasing power and technique.',
    'Surfing': 'A surfer riding a wave on a surfboard, displaying balance and skill.',
    'TableTennisShot': 'A player making a shot in table tennis, showing speed and precision.',
    'Typing': 'Someone typing on a keyboard, their fingers moving swiftly across the keys.',
    'UnevenBars': 'A gymnast performing a routine on the uneven bars, demonstrating agility and grace.',
    'WallPushups': 'An individual doing wall pushups, using a wall for resistance training.',
    'WritingOnBoard': 'A person writing on a board with a marker or chalk, conveying information.'
    }

import os
import pathlib
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import gc

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments
import clip
from vifiClip import AlignNet

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

        logits = model(batch)
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

            logits = model(batch)

            loss = criterion(logits, label)
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
                        choices=["TempNet", "VCLAPNet", "AlignNet"])
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
                                                                        image_processor, data.collate_fn, videomae_model, batch_size=32)
    
    # train_dataset, test_seen_dataset, test_unseen_dataset = data.get_default_dataset(dataset_root_path)
    # train_loader, test_seen_loader, test_unseen_loader = data.get_default_dataloader(train_dataset, test_seen_dataset, \
    #                                                                                  test_unseen_dataset, data.collate_fn, batch_size=16)
    
    
    # text features
    text_features = utils.get_text_features(device, encoder_choice=args.text_encoder)
    classname = list(utils.get_labels()[0].keys())

    # clip model
    clip_model, _ = clip.load("RN101", device)
    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)

    if args.network == "TempNet": model = transformerNet.TempNet(videomae_model, text_features, av_emb_size=768, device=device)
    elif args.network == "VCLAPNet": model = transformerNet.VCLAPNet(text_features, av_emb_size=512, device=device)
    elif args.network == "AlignNet": model = AlignNet(videomae_model, classname, clip_model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, train_accuracies, \
        test_seen_losses, test_seen_accuracies, \
            test_unseen_losses, test_unseen_accuracies = train(model, device, train_loader, test_seen_loader, test_unseen_loader, 
                                                                optimizer, criterion, scheduler, num_epoch=args.num_epoch)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")



    
