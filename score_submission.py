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


def dataframe_to_dict(df):
    mask_ids = df['img'].values
    rle_masks = df['rle_mask'].values
    id_to_mask = dict(zip(mask_ids, rle_masks))
    return mask_ids, id_to_mask


def score_from_csv(submission_csv, ground_truth_csv):
    submission_df = pd.read_csv(submission_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    score_from_dataframe(submission_df, ground_truth_df)
    pass


def score_from_dataframe(submission_df, ground_truth_df):
    scores = []

    _, id_to_prediction = dataframe_to_dict(submission_df)
    sample_ids, id_to_ground_truth = dataframe_to_dict(ground_truth_df)

    for mask_id in sample_ids:
        if mask_id not in id_to_prediction:
            raise ValueError("Missing id in prediction, can't compute score")
        prediction = rle_decode(id_to_prediction[mask_id], SIZE)
        ground_truth = rle_decode(id_to_ground_truth[mask_id], SIZE)
        score = dice_score(ground_truth, prediction)
        scores.append(score)

    print('Dice score for submission is {}'.format(np.mean(scores)))


if __name__ == "__main__":
    score_from_csv(args.submission_csv_path, args.ground_truth_csv_path)
