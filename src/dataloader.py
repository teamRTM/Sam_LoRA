import torch
import glob
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import pickle
import random


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt

    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(
        self,
        annotations: dict,
        processor: Samprocessor,
        mode: str,
        gt_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__()
        random.seed(random_seed)
        self.img_files, self.evaluate_files, self.train_files = [], [], []
        self.gt_ratio = gt_ratio
        if mode == "train":
            modes = ["train"]
        elif mode == "test":
            modes = ["test"]
        elif mode == "all":
            modes = ["train", "test"]

        # Get the image and mask path
        for data_mode in modes:
            for img_path, value in annotations[data_mode].items():
                self.img_files.append(img_path)
                self.evaluate_files.append(value["gt_path"])
                self.train_files.append(value["uad_pred_path"])

        # Replace train files with ground truth via gt_ratio
        if (mode == "train") and (gt_ratio > 0):
            n_data = len(self.img_files)
            n_gt_train = int(n_data * gt_ratio)
            replaced_idx = random.sample(range(n_data), n_gt_train)
            for idx in replaced_idx:
                self.train_files[idx] = self.evaluate_files[idx]

        # Get the anomaly map of UAD 
        self.uad_anomaly_maps = []
        for data_mode in modes:     
            for img_path, value in annotations[data_mode].items():
                self.uad_anomaly_maps.append(value["uad_pred_path"].replace("images", "anomaly_maps").replace(".png", ".pickle"))

        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int) -> list:
        img_path = self.img_files[index]
        train_mask_path = self.train_files[index]
        evaluate_mask_path = self.evaluate_files[index]

        # get image and train mask in PIL format
        image = Image.open(img_path)
        train_mask = Image.open(train_mask_path)
        train_mask = train_mask.convert("1")
        train_mask = np.array(train_mask)
        original_size = tuple(image.size)[::-1]

        # get bounding box prompt
        box = utils.get_bounding_box(train_mask)
        inputs = self.processor(image, original_size, box)
        inputs["train_mask"] = torch.from_numpy(train_mask)

        # add evaluate mask
        evaluate_mask = Image.open(evaluate_mask_path)
        evaluate_mask = evaluate_mask.convert("1")
        evaluate_mask = np.array(evaluate_mask)
        inputs["evaluate_mask"] = torch.from_numpy(evaluate_mask)

        # get anomaly map if test
        if len(self.uad_anomaly_maps) > 0:
            with open(self.uad_anomaly_maps[index], "rb") as f:
                uad_anomaly_map = pickle.load(f)
                f.close()
            inputs["anomaly_map"] = uad_anomaly_map

        return inputs


def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)
