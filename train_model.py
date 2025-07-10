#!/usr/bin/env python
"""
Fine‑tune Mask2Former on your custom semantic segmentation dataset.

Directory layout (example):

dataset/
├── train/
│   ├── images/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── masks/
│       ├── 0001.png
│       └── ...
└── val/
    ├── images/
    │   ├── 0421.jpg
    │   └── ...
    └── masks/
        ├── 0421.png
        └── ...

Each *mask* PNG must have the same file name as its image and store class‑ids
(0 … num_classes‑1) encoded as integer pixel values.

Usage:

$ pip install torch torchvision --upgrade
$ pip install "transformers[torch]" accelerate datasets pillow albumentations==1.3.1
$ python train_mask2former.py --data_root ./dataset --num_classes 21

"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torchvision
from tqdm import tqdm
import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (Mask2FormerForUniversalSegmentation,
                          Mask2FormerImageProcessor, Trainer,
                          TrainingArguments, default_data_collator)


def visualize_prediction(image, mask, pred_mask):
    """Visualizes the original image, ground truth mask, and predicted mask."""
    image = image + np.min(image)
    image = image / np.max(image) 

    mask = np.logical_not(mask)  # Invert mask to visualize as white on black
    pred_mask = np.logical_not(pred_mask)  # Invert predicted mask
    image[:,:,0] = image[:, :, 0] + mask 
    image[:,:,1] = image[:, :, 1] + pred_mask 
    image = image / np.max(image)  # Normalize to [0, 1] for visualization
    # Convert to uint8 for visualization
    image = (image * 255).astype(np.uint8)
    #save img with PIL
   # maskvis = np.zeros_like(image) 
   # maskvis[:,:,0] = mask*255

    img = Image.fromarray(image)
    img.save("image.png")

def get_file_pairs(split_dir: Path) -> List[Tuple[Path, Path]]:
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*")):
        mask_path = mask_dir / (img_path.stem + ".png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


class SegmentationDataset(Dataset):
    """Loads (image, segmentation mask) pairs and applies preprocessing."""

    def __init__(
        self,
        image_processor: Mask2FormerImageProcessor,
        train = True,
        aug = False,
        path: str = "data/",
    ):
        self.images = []
        self.masks = []
        for file in os.listdir(path + "images"):
            if file.endswith('.png'):
                # Assuming the mask has the same name as the image but in the masks directory
                self.images.append(os.path.join(path+ "images", file))
                self.masks.append(os.path.join(path+ "masks", file))
        self.file_pairs = list(zip(self.images, self.masks))
        if train:
            self.file_pairs = self.file_pairs[:int(len(self.file_pairs) * 0.8)]
        else:
            self.file_pairs = self.file_pairs[int(len(self.file_pairs) * 0.8):]
        
        self.processor = image_processor
        if aug:
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(
                        height=image_processor.size["height"],
                        width=image_processor.size["width"],
                        scale=(0.5, 1.0),
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    A.Rotate(limit=30, p=0.5),
                ],
                additional_targets={"mask": "mask"},
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height=image_processor.size["height"],
                        width=image_processor.size["width"],
                    )
                ],
                additional_targets={"mask": "mask"},
            )
        

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.file_pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        # Albumentations expects HWC images and HW masks
        augmented = self.transform(image=image, mask=mask)
        enc = self.processor(
            images=augmented["image"],
            segmentation_maps=augmented["mask"],
            return_tensors="pt",
        )
        #print(enc)
        # Remove batch dim added by processor
        enc = {k: v[0] for k, v in enc.items()}
        # The processor returns 'class_labels' (instance‑level). For semantic
        # segmentation we only need pixel_label_ids.
        #enc["labels"] = enc.pop("mask_labels")
        enc["image"] = augmented["image"]
        return enc


class custom_model(torch.nn.Module):
    def __init__(self, model_name, num_classes, processor):
        super(custom_model, self).__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.processor = processor
        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values).pixel_decoder_last_hidden_state
        outputs = self.linear(outputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outputs = torchvision.transforms.Resize(pixel_values.shape[2:])(outputs)
        return outputs

def parse_args():
    ap = argparse.ArgumentParser()
   # ap.add_argument("--data_root", type=Path, default = "data/", help="Dataset root dir")
    ap.add_argument("--model_name", type=str, default="facebook/mask2former-swin-base-IN21k-ade-semantic")
    ap.add_argument("--output_dir", type=Path, default="./checkpoints")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=25)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_classes", type=int, default = 2)
    return ap.parse_args()



def train_one_epoch(model, dataloader, optimizer, scaler, device, ignore_index):
    model.train()
    running_loss, running_acc_pixels, running_acc_correct = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    loop = tqdm(dataloader, leave=False, desc="Train")
    for batch in loop:
        pixel_values = batch["pixel_values"].to(device)

        mask_labels = batch["mask_labels"].squeeze(1).long().to(device)

      #  print(mask_labels)

       # print(batch["image"][0].shape)
       # visualize_prediction(batch["image"][0].cpu().numpy(), mask_labels[0].cpu().numpy(), np.zeros_like(mask_labels[0].cpu().numpy()))

        with torch.cuda.amp.autocast(enabled=scaler is not None):
    

            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs, mask_labels)  # CrossEntropyLoss expects (N, C, H, W) and (N, H, W)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * pixel_values.size(0)

        preds = outputs.argmax(1)
        running_acc_correct += (preds == mask_labels).sum().item()
        running_acc_pixels += mask_labels.shape[0] * mask_labels.shape[1] * mask_labels.shape[2]

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc = running_acc_correct / running_acc_pixels if running_acc_pixels else 0
    return avg_loss, avg_acc


def evaluate(model, dataloader, device, ignore_index):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    running_loss, running_acc_pixels, running_acc_correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="Val"):
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = batch["mask_labels"].squeeze(1).long().to(device)
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs, mask_labels)  
            running_loss += loss.item() * pixel_values.size(0)

            preds = outputs.argmax(1)
            mask = mask_labels != ignore_index
            running_acc_correct += (preds == mask_labels) .sum().item()
            running_acc_pixels += mask_labels.shape[0] * mask_labels.shape[1] * mask_labels.shape[2]

            #visualize some random prediction 
        random_label = np.random.randint(0, mask_labels.shape[0])
        visualize_prediction(batch["image"][random_label].cpu().numpy(), mask_labels[random_label].cpu().numpy(), preds[random_label].cpu().numpy())


    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc = running_acc_correct / running_acc_pixels if running_acc_pixels else 0
    return avg_loss, avg_acc

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ignore_index = 255  # Default ignore index for segmentation tasks

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load processor & model ------------------------------------------------
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_name, reduce_labels=False)
    model = custom_model(args.model_name, args.num_classes, processor).to(args.device)


    image_processor = Mask2FormerImageProcessor.from_pretrained(
        args.model_name,
        reduce_labels=False,  # Keep 0 … num_classes‑1
    )

    train_ds = SegmentationDataset( image_processor,train = True, aug=True)
    val_ds = SegmentationDataset( image_processor, train = False, aug=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=4,
    )

  
      # 3. Optimiser & scaler ----------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.device.startswith("cuda") else None

    # 4. Training loop ---------------------------------------------------------
    best_val_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, args.device, args.ignore_index)
        print(f"Train   | loss: {train_loss:.4f}  acc: {train_acc:.4f}")

       
        val_loss, val_acc = evaluate(model, val_loader, args.device, args.ignore_index)
        print(f"Val     | loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = args.output_dir / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "processor": processor.to_json_string(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"✅ Saved new best model to {ckpt_path} (acc={val_acc:.4f})")

    print(f"Training finished. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
