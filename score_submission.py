import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import rle_decode, dice_score

SIZE = (720, 1280)
parser = argparse.ArgumentParser(description='Scores a csv submission')

parser.add_argument("--submission_csv_path", help="Submission path", required=True)
parser.add_argument("--ground_truth_csv_path", help="Ground truth submission path", required=True)
args = parser.parse_args()


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    mask_ids = df['img'].values
    rle_masks = df['rle_mask'].values
    id_to_mask = dict(zip(mask_ids, rle_masks))
    return mask_ids, id_to_mask


if __name__ == "__main__":
    scores = []

    _, id_to_prediction = read_csv(args.submission_csv_path)
    sample_ids, id_to_ground_truth = read_csv(args.ground_truth_csv_path)

    for mask_id in sample_ids:
        if mask_id not in id_to_prediction:
            raise ValueError("Missing id in prediction, can't compute score")
        prediction = rle_decode(id_to_prediction[mask_id], SIZE)
        ground_truth = rle_decode(id_to_ground_truth[mask_id], SIZE)
        score = dice_score(ground_truth, prediction)
        scores.append(score)

    print('Dice score for submission is {}'.format(np.mean(scores)))

