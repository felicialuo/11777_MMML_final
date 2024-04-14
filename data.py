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
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data

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
    label2id, id2label = utils.get_labels(dataset_root_path)
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        if k == "video" or k == "audio":
            print(k, sample_video[k].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video['label']]}")


def get_dataset(dataset_root_path):

    label2id, id2label = utils.get_labels(dataset_root_path)


    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 8
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps


    # Training dataset transformations.
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
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
    )

    # Test seen and unseen datasets' transformations.
    test_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
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
    )

    test_unseen_dataset = Ucf101(
        data_path=os.path.join(dataset_root_path, "test_unseen"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        map_label2id=label2id,
        decode_audio=True,
        transform=test_transform,
    )

    return train_dataset, test_seen_dataset, test_unseen_dataset



if __name__ == "__main__":
    #test load dataset
    dataset_root_path = 'datasets/UCF101/UCF-101/'
    train_dataset, test_seen_dataset, test_unseen_dataset = get_dataset(dataset_root_path)
    print(train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)

    #test investigate_video
    sample_video = next(iter(test_seen_dataset))
    investigate_video(sample_video)



    
