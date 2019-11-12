import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import rle_decode, dice_score

SIZE = (720, 1280)
parser = argparse.ArgumentParser(description='Scores a csv submission')

parser.add_argument("--ground_truth_folder", help="Folder containing png masks with ground truth", required=True)
parser.add_argument("--submission_csv_path", help="Submission path", required=True)
parser.add_argument("--sample_csv_path", help="Sample submission path", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    scores = []

    submission_df = pd.read_csv(args.submission_csv_path)
    mask_ids = submission_df['img'].values
    rle_masks = submission_df['rle_mask'].values
    id_to_mask = dict(zip(mask_ids, rle_masks))

    sample_df = pd.read_csv(args.sample_csv_path)
    sample_ids = sample_df['img'].values
    for mask_id in sample_ids:
        mask_path = os.path.join(args.ground_truth_folder, '{}.png'.format(mask_id))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        prediction = rle_decode(id_to_mask[mask_id], SIZE)
        score = dice_score(mask, prediction)
        scores.append(score)

    print('Dice score for submission is {}'.format(np.mean(scores)))

