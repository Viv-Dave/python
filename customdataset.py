from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import json

class CC50KDataset(Dataset):
    def __init__(self, root_dir, processor=None):
        self.root_dir = root_dir

        # Correct file paths
        self.metadata_path = os.path.join(root_dir, "metadata.csv")
        self.image_dir = os.path.join(root_dir, "images")

        # Load CSV as DataFrame (this was missing before)
        self.metadata = pd.read_csv(self.metadata_path)

        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        filename = row["filename"]   
        caption = row["caption"]

        image_path = os.path.join(self.image_dir, filename)

        # Load the image
        image = Image.open(image_path).convert("RGB")

        if self.processor:
            return self.processor(images=image, text=caption, return_tensors="pt")

        return image, caption

class COCODataset(Dataset):
    def __init__(self, image_dir, caption_json, processor=None):
        """
        image_dir: directory containing COCO images (e.g., train2014/)
        caption_json: COCO annotation file (e.g., captions_train2014.json)
        processor: optional HuggingFace processor (BLIP, CLIP, etc.)
        """

        self.image_dir = image_dir
        self.processor = processor


        with open(caption_json, "r") as f:
            data = json.load(f)

        self.annotations = data["annotations"]
        images = {img["id"]: img["file_name"] for img in data["images"]}
        self.image_lookup = images

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        caption = ann["caption"]
        image_id = ann["image_id"]
        file_name = self.image_lookup[image_id]

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.processor:
            return self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )

        # Raw return
        return image, caption

class Flickr30kDataset(Dataset):
    def __init__(self, hf_dataset, processor=None, split=None):
        """
        hf_dataset : the loaded HF dataset (load_from_disk)
        processor  : image-text processor (e.g., CLIPProcessor)
        split      : 'train', 'val', 'test', or None (use entire dataset)
        """

        # Filter by split if provided
        if split is not None:
            self.dataset = hf_dataset.filter(lambda x: x["split"] == split)
        else:
            self.dataset = hf_dataset

        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        image = example["image"]                  # PIL object
        caption = example["alt_text"]             # caption field

        # Apply processor if provided
        if self.processor:
            processed = self.processor(
                images=image,
                text=caption,
                return_tensors="pt"
            )
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "caption": caption,
                "img_id": example["img_id"],
            }

        # Return raw if processor unavailable
        return image, caption
