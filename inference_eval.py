import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.metrics import calculate_pixel_auroc_from_logits, calculate_best_pixel_f1_score, calculate_iou
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import json
import torch.nn.functional as F
import monai
import numpy as np

"""
This file compute the evaluation metric (Dice cross entropy loss) for all trained LoRA SAM with different ranks. This gives the plot that is in ./plots/rank_comparison.jpg
which compares the performances on test the test set.
"""

# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    ymlfile.close()

# Load the annotations
with open("./annotations.json", "r") as jsonfile:
    annotations = json.load(jsonfile)
    jsonfile.close()

# Load the anomalib's config file
with open("../anomalib/src/anomalib/models/patchcore/config.yaml", "r") as ymlfile:
    anomalib_config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    ymlfile.close()

device = f"cuda:{config_file['TRAIN']['CUDA']}" if torch.cuda.is_available() else "cpu"
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Get the category
categroy = anomalib_config_file["dataset"]["category"]

# Get rank list
rank_list = config_file["SAM"]["RANKS"]

rank_loss = []
rank_pixel_auroc = []
rank_pixel_f1_score = []
rank_miou = []

total_baseline_loss = []
total_baseline_pixel_auroc = []
total_baseline_pixel_f1_score = []
total_baseline_iou = []

total_uad_pixel_auroc = []
total_uad_pixel_f1_score = []
total_uad_iou = []
baseline_loss = 0

# Load SAM model
with torch.no_grad():
    # ------------------- Baseline evaluation ------------------- #
    print("Base model evaluation...")
    sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
    baseline = sam
    processor = Samprocessor(baseline)
    dataset = DatasetSegmentation(annotations, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    baseline.eval()
    baseline.to(device)
    for i, batch in enumerate(tqdm(test_dataloader)):
        outputs = baseline(batched_input=batch, multimask_output=False)

        gt_mask_tensor = (
            batch[0]["evaluate_mask"].unsqueeze(0).unsqueeze(0)
        )  # We need to get the [B, C, H, W] starting from [H, W]
        loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))

        # Calculate Pixel AUROC
        pixel_auroc = calculate_pixel_auroc_from_logits(
            gt_mask_tensor.cpu().numpy(), outputs[0]["low_res_logits"].cpu().numpy()
        )

        # Calculate pixel F1 score
        pixel_f1_score, _, _, _ = calculate_best_pixel_f1_score(
            gt_mask_tensor.cpu().numpy(), outputs[0]["low_res_logits"].cpu().numpy()
        )

        # Calculate IOU
        iou = calculate_iou(
            gt_mask_tensor.cpu().numpy(), outputs[0]["masks"].cpu().numpy()
        )

        total_baseline_loss.append(loss.item())
        total_baseline_pixel_auroc.append(pixel_auroc)
        total_baseline_pixel_f1_score.append(pixel_f1_score)
        total_baseline_iou.append(iou)

    print(f"Mean dice score: {mean(total_baseline_loss)}")
    print(f"Mean Pixel AUROC score: {mean(total_baseline_pixel_auroc)}")
    print(f"Mean pixel F1 score: {mean(total_baseline_pixel_f1_score)}")
    print(f"Mean IOU score: {mean(total_baseline_iou)}")
    baseline_loss = mean(total_baseline_loss)
    baseline_pixel_auroc = mean(total_baseline_pixel_auroc)
    baseline_pixel_f1_score = mean(total_baseline_pixel_f1_score)
    baseline_miou = mean(total_baseline_iou)

    # ------------------- PatchCore evaluation ------------------- #
    print("\n")
    print("UAD evaluation...")
    for i, batch in enumerate(tqdm(test_dataloader)):
        gt_mask_tensor = batch[0]["evaluate_mask"].unsqueeze(0).unsqueeze(0)
        uad_mask_tensor = batch[0]["train_mask"].unsqueeze(0).unsqueeze(0)
        uad_anoamly_map = torch.tensor(batch[0]["anomaly_map"]).unsqueeze(0)

        # Uad interpolate for gt mask size 
        uad_anoamly_map = F.interpolate(uad_anoamly_map, size=gt_mask_tensor.shape[2:])
        uad_anoamly_map = uad_anoamly_map.numpy()

        # Calculate Pixel AUROC
        pixel_auroc = calculate_pixel_auroc_from_logits(
            gt_mask_tensor.cpu().numpy(), uad_anoamly_map
        )

        # Calculate pixel F1 score
        pixel_f1_score, _, _, _ = calculate_best_pixel_f1_score(
            gt_mask_tensor.cpu().numpy(), uad_anoamly_map
        )

        # Calculate IOU
        iou = calculate_iou(
            gt_mask_tensor.cpu().numpy(), uad_mask_tensor.cpu().numpy()
        )

        total_uad_pixel_auroc.append(pixel_auroc)
        total_uad_pixel_f1_score.append(pixel_f1_score)
        total_uad_iou.append(iou)
    
    print(f"Mean Pixel AUROC score: {mean(total_uad_pixel_auroc)}")
    print(f"Mean pixel F1 score: {mean(total_uad_pixel_f1_score)}")
    print(f"Mean IOU score: {mean(total_uad_iou)}")
    uad_pixel_auroc = mean(total_uad_pixel_auroc)
    uad_pixel_f1_score = mean(total_uad_pixel_f1_score)
    uad_miou = mean(total_uad_iou)

    # ------------------- LoRA evaluation ------------------- #
    for rank in rank_list:
        print("\n")
        print(f"Rank {rank} evaluation...")
        sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
        baseline = sam

        # Create SAM LoRA
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(
            f"../results/sam_lora_weights/mvtec_{categroy}_rank{rank}.safetensors"
        )
        model = sam_lora.sam

        # Process the dataset
        processor = Samprocessor(model)
        dataset = DatasetSegmentation(annotations, processor, mode="test")

        # Create a dataloader
        test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        # Set model to train and into the device
        model.eval()
        model.to(device)

        total_score = []
        total_pixel_auroc = []
        total_pixel_f1_score = []
        total_iou = []
        for i, batch in enumerate(tqdm(test_dataloader)):
            outputs = model(batched_input=batch, multimask_output=False)

            gt_mask_tensor = (
                batch[0]["evaluate_mask"].unsqueeze(0).unsqueeze(0)
            )  # We need to get the [B, C, H, W] starting from [H, W]
            loss = seg_loss(
                outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device)
            )

            # Calculate Pixel AUROC
            pixel_auroc = calculate_pixel_auroc_from_logits(
                gt_mask_tensor.cpu().numpy(), outputs[0]["low_res_logits"].cpu().numpy()
            )

            # Calculate pixel F1 score
            pixel_f1_score, _, _, _ = calculate_best_pixel_f1_score(
                gt_mask_tensor.cpu().numpy(), outputs[0]["low_res_logits"].cpu().numpy()
            )

            # Calculate IOU
            iou = calculate_iou(
                gt_mask_tensor.cpu().numpy(), outputs[0]["masks"].cpu().numpy()
            )

            total_score.append(loss.item())
            total_pixel_auroc.append(pixel_auroc)
            total_pixel_f1_score.append(pixel_f1_score)
            total_iou.append(iou)

        print(f"Mean dice score: {mean(total_score)}")
        print(f"Mean Pixel AUROC score: {mean(total_pixel_auroc)}")
        print(f"Mean pixel F1 score: {mean(total_pixel_f1_score)}")
        print(f"Mean IOU score: {mean(total_iou)}")
        rank_loss.append(mean(total_score))
        rank_pixel_auroc.append(mean(total_pixel_auroc))
        rank_pixel_f1_score.append(mean(total_pixel_f1_score))
        rank_miou.append(mean(total_iou))

print("RANK LOSS :", rank_loss)
print("RANK Pixel AUROC :", rank_pixel_auroc)
print("RANK PIXEL F1 SCORE :", rank_pixel_f1_score)
print("RANK MIOU :", rank_miou)

# ------------------- Plot loss graph ------------------- #
width = 0.25  # the width of the bars
multiplier = 0
models_loss_results = {
    "Baseline": baseline_loss,
    "Rank 4": rank_loss[0],
    "Rank 8": rank_loss[1],
    "Rank 16": rank_loss[2],
    "Rank 64": rank_loss[3],
    "Rank 128": rank_loss[4],
    "Rank 256": rank_loss[5],
    "Rank 512": rank_loss[6],
    "Rank 1024": rank_loss[7],
}

eval_scores_name = ["Rank"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout="constrained")

for model_name, score in models_loss_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Dice Loss")
ax.set_title("LoRA trained on 50 epochs - Rank comparison on test set")
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=1, ncols=2)
ax.set_ylim(0, 0.5)

plt.savefig("../results/final_results/rank_loss_comparison.jpg")

# ------------------- Plot Pixel AUROC graph ------------------- #

width = 0.25  # the width of the bars
multiplier = 0
models_pixel_auroc_results = {
    "Baseline": baseline_pixel_auroc,
    "UAD": uad_pixel_auroc,
    "Rank 4": rank_pixel_auroc[0],
    "Rank 8": rank_pixel_auroc[1],
    "Rank 16": rank_pixel_auroc[2],
    "Rank 64": rank_pixel_auroc[3],
    "Rank 128": rank_pixel_auroc[4],
    "Rank 256": rank_pixel_auroc[5],
    "Rank 512": rank_pixel_auroc[6],
    "Rank 1024": rank_pixel_auroc[7],
}

eval_scores_name = ["Rank"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout="constrained")

for model_name, score in models_pixel_auroc_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("PIXEL AUROC")
ax.set_title("LoRA trained on 50 epochs - Rank comparison on test set")
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=1, ncols=2)
ax.set_ylim(0.9, 1.05)

plt.savefig("../results/final_results/rank_pixel_auroc_comparison.jpg")

# ------------------- Plot Pixel F1 score graph ------------------- #

width = 0.25  # the width of the bars
multiplier = 0
models_pixel_f1_score_results = {
    "Baseline": baseline_pixel_f1_score,
    "UAD": uad_pixel_f1_score,
    "Rank 4": rank_pixel_f1_score[0],
    "Rank 8": rank_pixel_f1_score[1],
    "Rank 16": rank_pixel_f1_score[2],
    "Rank 64": rank_pixel_f1_score[3],
    "Rank 128": rank_pixel_f1_score[4],
    "Rank 256": rank_pixel_f1_score[5],
    "Rank 512": rank_pixel_f1_score[6],
    "Rank 1024": rank_pixel_f1_score[7],
}

eval_scores_name = ["Rank"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout="constrained")

for model_name, score in models_pixel_f1_score_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("PIXEL F1 SCORE")
ax.set_title("LoRA trained on 50 epochs - Rank comparison on test set")
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=1, ncols=2)
ax.set_ylim(0.6, 0.9)

plt.savefig("../results/final_results/rank_pixel_f1_score_comparison.jpg")

# ------------------- Plot Pixel mIoU graph ------------------- #

width = 0.25  # the width of the bars
multiplier = 0
models_miou_results = {
    "Baseline": baseline_miou,
    "UAD": uad_miou,
    "Rank 4": rank_miou[0],
    "Rank 8": rank_miou[1],
    "Rank 16": rank_miou[2],
    "Rank 64": rank_miou[3],
    "Rank 128": rank_miou[4],
    "Rank 256": rank_miou[5],
    "Rank 512": rank_miou[6],
    "Rank 1024": rank_miou[7],
}

eval_scores_name = ["Rank"]
x = np.arange(len(eval_scores_name))
fig, ax = plt.subplots(layout="constrained")

for model_name, score in models_miou_results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model_name)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("MIOU")
ax.set_title("LoRA trained on 50 epochs - Rank comparison on test set")
ax.set_xticks(x + width, eval_scores_name)
ax.legend(loc=1, ncols=2)
ax.set_ylim(0.45, 0.63)

plt.savefig("../results/final_results/rank_miou_comparison.jpg")