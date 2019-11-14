# starter-challenge-data-2020

# Info
This starter contains:
- helper functions to convert to RLE (Run-length encoding) and back
- a scoring function for Dice Score
- a starter project using Keras to generate predictions

# Requirements
- Python 3.5+
- `numpy==1.16.1`
- `opencv-python==3.4.5.20`
- `pandas==0.23.4`

# Helpers

### `make_rle_submission.py`
Usage:

    python make_rle_submission.py --mask_folder ../data_challenge/output --output_path submission.csv --sample_csv_path sample_submission.csv

Note that the `mask_folder` must only contain .png files with binary masks (values in [0, 1] or [0, 255])    
  
### `score_submission.py`
Usage:

    python score_submission.py --submission_csv_path submission.csv --ground_truth_csv_path ./ground_truth.csv

# Util functions
### RLE decode
    from utils improt rle_encode, rle_to_string
    
    rle_runs = rle_encode(mask)
    rle_string = rle_to_string(rle_runs)
    
### RLE encode
    from utils improt rle_encode, rle_to_string
    
    size = (720, 1280)
    mask = rle_decode(rle_string, (720, 1280))

### Dice score
    from utils import dice_score

    score = dice_score(grount_truth, prediction)
    
# Starter project
Heavily inspired from the Kaggle Carvana's [third place solution](https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/), it is a simple Keras sample allowing competitors to get started. Get started with `benchmark.ipynb`.
