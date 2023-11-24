import torch
import numpy as np
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image, ImageDraw
import yaml
import json
from pathlib import Path
from torchvision.transforms import ToTensor

"""
This file is used to plots the predictions of a model (either baseline or LoRA) on the train or test set. Most of it is hard coded so I would like to explain some parameters to change 
referencing by lines : 
line 22: change the rank of lora; line 98: Do inference on train (inference_train=True) else on test; line 101 and 111 is_baseline arguments in fuction: True to use baseline False to use LoRA model. 
"""

# Load Sam LoRA configs
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    ymlfile.close()

# Load anomalib's config file
with open("../anomalib/src/anomalib/models/patchcore/config.yaml", "r") as ymlfile:
    anomalib_config = yaml.load(ymlfile, Loader=yaml.Loader)
    ymlfile.close()

# Open annotation file
f = open("annotations.json")
annotations = json.load(f)

# Set sam checkpoint
sam_checkpoint = "../sam_weights/sam_vit_b_01ec64.pth"
device = f"cuda:{config_file['TRAIN']['CUDA']}" if torch.cuda.is_available() else "cpu"


def inference_model(
    sam_model,
    image_path,
    save_path,
    rank,
    mask_path=None,
    bbox=None,
    is_baseline=False,
):
    filename = f"{save_path.parent.name}_{save_path.name}"

    if is_baseline == False:
        model = sam_model.sam
        rank = sam_model.rank
    else:
        model = build_sam_vit_b(checkpoint=sam_checkpoint)

    model.eval()
    model.to(device)
    image = Image.open(image_path)
    if mask_path != None:
        mask = Image.open(mask_path)
        mask = mask.convert("1")
        ground_truth_mask = np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
    else:
        box = bbox

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False,
    )

    if mask_path == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(masks[0])
        if is_baseline:
            ax2.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(save_path)
        else:
            ax2.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(save_path)

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharex=True, sharey=True, figsize=(15, 15)
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")

        ax3.imshow(masks[0])
        if is_baseline:
            ax3.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(save_path)
        else:
            ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(save_path)
    plt.close()


# Get ranks from config file
ranks = config_file["SAM"]["RANKS"]

# Base line visualization
# Set dst
dst = Path(f"../results/final_results/baseline")
dst.mkdir(parents=True, exist_ok=True)

# Get save file name
train_set = annotations["train"]
test_set = annotations["test"]

# Train data inference
for image_name, dict_annot in train_set.items():
    image_name = Path(image_name)
    save_path = dst / image_name.parent.name / image_name.name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    inference_model(
        None,
        image_name,
        save_path,
        rank=4,
        mask_path=dict_annot["gt_path"],
        bbox=dict_annot["bbox"],
        is_baseline=True,
    )

# Test data inference
for image_name, dict_annot in test_set.items():
    image_name = Path(image_name)
    save_path = dst / image_name.parent.name / image_name.name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    inference_model(
        None,
        image_name,
        save_path,
        rank=4,
        mask_path=dict_annot["gt_path"],
        bbox=dict_annot["bbox"],
        is_baseline=True,
    )

# Trained model visualization
for rank in ranks:
    print(f"Rank {rank} plotting...")
    # Set dst
    dst = Path(f"../results/final_results/{rank}")

    # Get save file name
    weight_src = Path("../results/sam_lora_weights")
    weight_name = (
        f"mvtec_{anomalib_config['dataset']['category']}_rank{rank}.safetensors"
    )

    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(str(weight_src / weight_name))
    model = sam_lora.sam

    train_set = annotations["train"]
    test_set = annotations["test"]
    inference_train = True

    # Train data inference
    for image_name, dict_annot in train_set.items():
        image_name = Path(image_name)
        save_path = dst / image_name.parent.name / image_name.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        inference_model(
            sam_lora,
            image_name,
            save_path,
            rank=rank,
            mask_path=dict_annot["gt_path"],
            bbox=dict_annot["bbox"],
            is_baseline=False,
        )

    # Test data inference
    for image_name, dict_annot in test_set.items():
        image_name = Path(image_name)
        save_path = dst / image_name.parent.name / image_name.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        inference_model(
            sam_lora,
            image_name,
            save_path,
            rank=rank,
            mask_path=dict_annot["gt_path"],
            bbox=dict_annot["bbox"],
            is_baseline=False,
        )
