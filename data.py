import os
import pathlib
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.clip_sampling import ClipSampler
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
import pytorchvideo.data
from torch.utils.data import DistributedSampler
import torchvision

import utils

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

class ZS_LabeledVideoDataset(pytorchvideo.data.LabeledVideoDataset):
    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        map_label2id: Dict[str, int],
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        super().__init__(
            labeled_video_paths,
            clip_sampler,
            video_sampler,
            transform,
            decode_audio,
            decoder,
        )
        self.map_label2id = map_label2id


    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>, eg.torch.Size([3, 16, 224, 224]),
                    'video_name': <file_name_str>, eg. v_FrontCrawl_g04_c06.avi,
                    'label': <index_label>, eg. 21,
                    'video_index': <video_index>, eg. 2054,
                    'clip_index': <clip_index>, eg. 0,
                    'aug_index': <aug_index>, eg. 0
                    **({"audio": audio_samples} if audio_samples is not None else {}), eg. torch.Size([93312])
                }
        """
        sample_dict = super().__next__()
        

        # Map the label to its corresponding ID of all seen and unseen classes
        class_name = sample_dict['video_name'].split('_')[1]

        sample_dict['label'] = self.map_label2id[class_name]

        return sample_dict


def ZS_labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    map_label2id: Dict[str, int],
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> ZS_LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.
    adapted from https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/data/labeled_video_dataset.html
    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = ZS_LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        map_label2id,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

def Ucf101(
    data_path: str,
    clip_sampler: ClipSampler,
    map_label2id: Dict[str, int],
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> ZS_LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for the Ucf101 dataset.
    adapted from https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/data/ucf101.html
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Ucf101")

    return ZS_labeled_video_dataset(
        data_path,
        clip_sampler,
        map_label2id,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

def investigate_video(sample_video):
    label2id, id2label = utils.get_labels() #dataset_root_path
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        if k == "video" or k == "audio":
            print(k, sample_video[k].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video['label']]}")


def _lambda_function(x):
    return x / 255.0

def get_iterable_dataset(dataset_root_path, image_processor, num_frames_to_sample=16,
                sample_rate=8, fps=30):

    label2id, id2label = utils.get_labels() #dataset_root_path
 

    # videomae_ckpt = "MCG-NJU/videomae-base"
    # image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)
    # model = VideoMAEForVideoClassification.from_pretrained(
    # videomae_ckpt,
    # label2id=label2id,
    # id2label=id2label,
    # ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    # )

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    # num_frames_to_sample = model.config.num_frames
    # sample_rate = 8
    # fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps


    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(_lambda_function),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    # Training dataset.
    train_dataset = Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        map_label2id=label2id,
        decode_audio=True,
        transform=train_transform,
        #video_sampler=DistributedSampler,
    )

    # Test seen and unseen datasets' transformations.
    test_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(_lambda_function),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    
    # Test seen and unseen datasets.
    test_seen_dataset = Ucf101(
        data_path=os.path.join(dataset_root_path, "test_seen"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        map_label2id=label2id,
        decode_audio=True,
        transform=test_transform,
        #video_sampler=DistributedSampler,
    )

    test_unseen_dataset = Ucf101(
        data_path=os.path.join(dataset_root_path, "test_unseen"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        map_label2id=label2id,
        decode_audio=True,
        transform=test_transform,
        #video_sampler=DistributedSampler,
    )

    return train_dataset, test_seen_dataset, test_unseen_dataset

def _normalization(x):
    return x/255.0

def _permute(x):
    return x.permute(1, 0, 2, 3)

def get_default_dataset(dataset_root_path, frames_per_clip = 96, step_between_clips = 48):
    train_transform=Compose(
                [
                    #Lambda(_pre_permute),
                    UniformTemporalSubsample(32),
                    Lambda(_permute),
                    Lambda(_normalization),
                    Normalize(0.5, 0.5),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop((224, 224)),
                    RandomHorizontalFlip(p=0.5),
                ]
            )


    test_transform=Compose(
                    [
                        #Lambda(_pre_permute),
                        UniformTemporalSubsample(32),
                        Lambda(_permute),
                        Lambda(_normalization),
                        Normalize(0.5, 0.5),
                        Resize((224, 224)),
                    ]
                )
    
    train_dataset = torchvision.datasets.UCF101(
        root=dataset_root_path + "train", annotation_path=dataset_root_path + "UCF101TrainTestSplits-Kaggle/ucfTrainTestlist",
        frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=True, num_workers=12, output_format = "TCHW", transform=train_transform
    )

    test_seen_dataset = torchvision.datasets.UCF101(
        root=dataset_root_path + "test_seen", annotation_path=dataset_root_path + "UCF101TrainTestSplits-Kaggle/ucfTrainTestlist",
        frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=False, num_workers=12, output_format = "TCHW", transform=test_transform
    )

    test_unseen_dataset = torchvision.datasets.UCF101(
        root=dataset_root_path + "test_unseen", annotation_path=dataset_root_path + "UCF101TrainTestSplits-Kaggle/ucfTrainTestlist",
        frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=False, num_workers=12, output_format = "TCHW", transform=test_transform
    )

    train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos = len(train_dataset), len(test_seen_dataset), len(test_unseen_dataset)

    return train_dataset, test_seen_dataset, test_unseen_dataset


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    # audio = torch.stack(
    #     [example["audio"]for example in examples]
    # ) # currently of different length
    audio = torch.zeros(pixel_values.shape)

    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "audio": audio, "labels": labels}
    # return {"pixel_values": pixel_values}


def get_iterable_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, image_processor, collate_fn, model, batch_size=16):
    num_epochs = 10
    args = TrainingArguments(
        "new_videomae",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        tf32=True,
        #dataloader_num_workers = 8, 
        dataloader_pin_memory = True,
        #dataloader_drop_last = True,
        #local_rank = -1,
        #dataloader_persistent_workers = True
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_seen_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # predictions = trainer.predict(test_seen_dataset).predictions 
    # print(predictions)
    train_loader = trainer.get_train_dataloader()
    test_seen_loader = trainer.get_test_dataloader(test_seen_dataset)
    test_unseen_loader = trainer.get_test_dataloader(test_unseen_dataset)

    return train_loader, test_seen_loader, test_unseen_loader


def get_default_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, collate_fn, batch_size=16):
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers = 12,
    )

    test_seen_loader = torch.utils.data.DataLoader(
    test_seen_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers = 12,
    )

    test_unseen_loader = torch.utils.data.DataLoader(
    test_unseen_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers = 12,
    )

    return train_loader, test_seen_loader, test_unseen_loader

def test():
    import time
    start_time = time.time()

    #test load dataset
    dataset_root_path = './datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)

    train_dataset, test_seen_dataset, test_unseen_dataset = get_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)

    # train_loader = get_dataloader(train_dataset, mode='train', batch_size=16, num_workers=1, pin_memory=True)
    # for x, y in train_loader:
    #     print("train_loader")
    #     print(x.shape, y.shape)
    #     break

    #test investigate_video
    sample_video = next(iter(test_seen_dataset))
    investigate_video(sample_video)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")


if __name__ == "__main__":
    from transformers import VideoMAEModel, VideoMAEImageProcessor, Trainer, TrainingArguments
    from transformerNet import TempNet
    import utils
    #test()
    dataset_root_path = './datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"
    videomae_model = VideoMAEModel.from_pretrained(videomae_ckpt)
    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset, test_seen_dataset, test_unseen_dataset = get_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)

    batch_size = 6
    num_epochs = 10
    args = TrainingArguments(
        "new_videomae",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        tf32=True,
        #dataloader_num_workers = 1, 
        dataloader_pin_memory = True,
        #dataloader_drop_last = True,
        #local_rank = -1,
        #dataloader_persistent_workers = True
    )
    text_features = utils.get_text_features(device)

    my_model = TempNet(videomae_model, text_features, av_emb_size=768, device=device)
    

    trainer = Trainer(
        my_model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_seen_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    trainer.train()


    





    
