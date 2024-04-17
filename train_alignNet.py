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

def contrastive_loss(video_feat, text_feat, device = 'cpu'):
        # Assuming video_feat and text_feat are already normalized
        cosine_sim = torch.mm(video_feat, text_feat.t()).to(device)
        labels = torch.arange(cosine_sim.size(0)).to(device)
        loss_video = nn.CrossEntropyLoss()
        loss_text = nn.CrossEntropyLoss()
        loss = (loss_video(video_feat, labels) + loss_text(text_feat, labels))/2
        return loss

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
     

def train_one_epoch(model, train_loader, device, optimizer, epoch, id_mapping):

    seen_label2id, seen_id2label, unseen_label2id, unseen_id2label = utils.get_sep_seen_unseen_labels()

    model.train()
    total_loss = 0
    correct = 0
    total_num = 0
    total_batch = 0
    for i, batch in enumerate(tqdm(train_loader, total=train_loader.dataset.num_videos//train_loader.batch_size+1, position=0, leave=False, desc="Train Epoch {}".format(epoch))):
        video = batch['pixel_values'].permute(0, 2, 1, 3, 4).to(device)
        labels = batch['labels'].to(device)
        texts = torch.cat([clip.tokenize(f"a video of a {seen_id2label[c.cpu().item()]}")for c in labels]).to(device)
        optimizer.zero_grad()
        video_feat, text_feat = model(video, texts)
        loss = contrastive_loss(video_feat, text_feat, device = device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batch += 1
        total_num += labels.shape[0]

        if i % 10 == 0:
            tqdm.write('\tTrain Set Batch {}: Current loss: {:.4f}. '.format(
                        i, loss.item()))

    return total_loss / total_batch, 100 * correct / total_num

def eval(model, test_loader, device, epoch, id_mapping, seen=True):
    
    seen_label2id, seen_id2label, unseen_label2id, unseen_id2label = utils.get_sep_seen_unseen_labels()
    model.eval()

    if seen:
        texts = torch.cat([clip.tokenize(f"a video of a {seen_id2label[c]}")for c in seen_id2label]).to(device)
    else:
        texts = torch.cat([clip.tokenize(f"a video of a {unseen_id2label[c]}")for c in unseen_id2label]).to(device)

    total_loss = 0
    total_correct = 0
    total_batch = 0
    total_num = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=test_loader.dataset.num_videos//test_loader.batch_size+1, position=0, leave=False, desc="Test Epoch {}".format(epoch))):
            video = batch['pixel_values'].permute(0, 2, 1, 3, 4).to(device)
            labels = batch['labels'].to(device)

            
            
            video_feat, text_feat = model(video, texts)
            #loss = contrastive_loss(video_feat, text_feat)
            #total_loss += loss.item()
            similarity = (100.0 * video_feat @ text_feat.T).softmax(dim=-1)

            pred = similarity.argmax(dim=1, keepdim=True)
            if seen:
                pred = torch.tensor([id_mapping[pred[i].item()] for i in range(pred.shape[0])]).to(device)
            else:
                pred = torch.tensor([id_mapping[pred[i].item()] for i in range(pred.shape[0])]).to(device)
            correct = pred.eq(labels.view_as(pred)).sum().item()
            total_correct += correct

            total_batch += 1
            total_num += labels.shape[0]
    
    if seen:
        tqdm.write('\nSeen Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        total_loss / total_batch, total_correct, total_num,
        100. * total_correct / total_num))
    else:
        tqdm.write('\nUnseen Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        total_loss / total_batch, total_correct, total_num,
        100. * total_correct / total_num))
    


if __name__ == "__main__":

    # Example use
    dataset_root_path = './datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)
    videomae_model = VideoMAEModel.from_pretrained(videomae_ckpt)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset, test_seen_dataset, test_unseen_dataset = data.get_iterable_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)
    
    train_loader, test_seen_loader, test_unseen_loader = data.get_iterable_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, 
                                                                        image_processor, data.collate_fn, videomae_model, batch_size=64)
    
    del videomae_model
    torch.cuda.empty_cache()
    gc.collect()

    seen_id2all_id, unseen_id2all_id = utils.get_id2id_mapping()
        
    model = transformerNet.AlignNet(device = device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(seen_id2all_id, unseen_id2all_id)

    for e in range(10):
        print(f"Epoch {e}")
        train_one_epoch(model, train_loader, device, optimizer, e, seen_id2all_id)
        eval(model, test_seen_loader, device, e, seen_id2all_id, seen=True)
        eval(model, test_unseen_loader, device, e, unseen_id2all_id, seen=False)
    