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



def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    # return {"pixel_values": pixel_values}


def get_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, image_processor, collate_fn, model,
                   mode='train', batch_size=16):
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
        #dataloader_num_workers = 12, 
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
    



if __name__ == "__main__":
    import time
    start_time = time.time()



    dataset_root_path = '../datasets/UCF101/UCF-101/'
    videomae_ckpt = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(videomae_ckpt)
    model = VideoMAEModel.from_pretrained(videomae_ckpt)

    train_dataset, test_seen_dataset, test_unseen_dataset = data.get_dataset(dataset_root_path, image_processor,
                                                                        num_frames_to_sample=16, sample_rate=8, fps=30)
    print("datasets", train_dataset.num_videos, test_seen_dataset.num_videos, test_unseen_dataset.num_videos)
    
    get_dataloader(train_dataset, test_seen_dataset, test_unseen_dataset, image_processor, collate_fn, model,
                   mode='train', batch_size=16)




    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time is {elapsed_time} seconds.")



    
