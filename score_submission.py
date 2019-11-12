import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from utils import rle_decode, dice_score

SIZE = (720, 1280)
parser = argparse.ArgumentParser(description='Scores a csv submission')

parser.add_argument("--ground_truth_folder", help="Folder containing png masks with ground truth", required=True)
parser.add_argument("--submission_csv", help="Submission filename", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    scores = []

    with open(args.submission_csv) as f:
        lines = f.readlines()[1:]
    print('Found {} lines in submission'.format(len(lines)))

    for line in lines:
        idx, rle_string = line.rstrip().split(',')
        mask_path = os.path.join(args.ground_truth_folder, '{}.png'.format(idx))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        prediction = rle_decode(rle_string, SIZE)
        score = dice_score(mask, prediction)
        scores.append(score)

    print('Dice score for submission is {}'.format(np.mean(scores)))

