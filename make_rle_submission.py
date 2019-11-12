import argparse
import os

import cv2
import pandas as pd

from utils import rle_encode, rle_to_string


parser = argparse.ArgumentParser(description='Converts all png images from a folder into a submission')

parser.add_argument("--mask_folder", help="Folder containing png masks", required=True)
parser.add_argument("--output_path", help="Output path", required=True)
parser.add_argument("--sample_csv_path", help="Sample submission csv", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.sample_csv_path)
    mask_ids = df['img'].values

    encoded_strings = []
    for mask_id in mask_ids:
        mask_path = os.path.join(args.mask_folder, '{}.png'.format(mask_id))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        encoded = rle_encode(mask)
        encoded_string = rle_to_string(encoded)
        encoded_strings.append(encoded_string)
    df['rle_mask'] = encoded_strings
    df.to_csv(args.output_path, index=False)
    print('Wrote submission in  to {}'.format(args.output_path))
