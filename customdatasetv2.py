from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import json
import torch

# Define a global fixed length for text to ensure tensors stack correctly
MAX_LENGTH = 50 

class CC50KDataset(Dataset):
    def __init__(self, root_dir, processor=None):
        self.root_dir = root_dir
        self.metadata_path = os.path.join(root_dir, "metadata.csv")
        self.image_dir = os.path.join(root_dir, "images")
        self.metadata = pd.read_csv(self.metadata_path)
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row["filename"]   
        caption = str(row["caption"]) # Ensure it's a string to prevent tokenizer errors

        image_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            # Fallback for corrupted images (optional safety)
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.processor:
            processed = self.processor(
                images=image, 
                text=caption, 
                return_tensors="pt",
                padding="max_length", # Force padding
                truncation=True,      # Force truncation
                max_length=MAX_LENGTH # Set fixed length
            )
            
            # Remove the batch dimension [1, ...] -> [...]
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0)
            }

        return image, caption

class COCODataset(Dataset):
    def __init__(self, image_dir, caption_json, processor=None):
        self.image_dir = image_dir
        self.processor = processor

        with open(caption_json, "r") as f:
            data = json.load(f)

        self.annotations = data["annotations"]
        # Create lookup map
        self.image_lookup = {img["id"]: img["file_name"] for img in data["images"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        caption = str(ann["caption"])
        image_id = ann["image_id"]
        
        file_name = self.image_lookup[image_id]
        img_path = os.path.join(self.image_dir, file_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.processor:
            processed = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            )
            
            # Remove the batch dimension and return consistent keys
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0)
            }

        return image, caption

class Flickr30kDataset(Dataset):
    def __init__(self, hf_dataset, processor=None, split=None):
        if split is not None:
            self.dataset = hf_dataset.filter(lambda x: x["split"] == split)
        else:
            self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]      
        caption = str(example["alt_text"])

        if self.processor:
            processed = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length", # Added missing padding
                truncation=True,      # Added missing truncation
                max_length=MAX_LENGTH # Added missing max_length
            )
            
            # Return ONLY the keys present in other datasets to avoid collate errors
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                # Removed "caption" and "img_id" to match CC50K and COCO structure
            }

        return image, caption