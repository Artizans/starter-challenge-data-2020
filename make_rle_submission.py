import argparse
import os
from pathlib import Path
import glob

import cv2

from utils import rle_encode, rle_to_string


parser = argparse.ArgumentParser(description='Converts all png images from a folder into a submission')

parser.add_argument("--mask_folder", help="Folder containing png masks", required=True)
parser.add_argument("--output_file", help="Output filename", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    encoded_strings = []

    mask_paths = glob.glob(os.path.join(args.mask_folder, '*.png'))
    print('Found {} files in folder'.format(len(mask_paths)))

    with open(args.output_file, 'w') as f:
        f.write('img,rle_mask\n')
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            filename = Path(mask_path).name
            idx = filename.replace('.png', '')
            encoded = rle_encode(mask)
            encoded_string = rle_to_string(encoded)
            f.write('{},{}\n'.format(idx, encoded_string))
