# starter-challenge-data-2020

# Info
This starter contains:
- helper functions to convert to RLE (Run-length encoding) and back
- a scoring function for Dice Score
- a starter project using Keras to generate predictions


# Helpers

### `make_rle_submission.py`
Usage:

    python make_rle_submission.py --mask_folder ./output --output_file submission.csv

Note that the `mask_folder` must only contain .png files with binary masks (values in [0, 1] or [0, 255])    
  
### `score_submission.py`
Usage:

    python score_submission.py --ground_truth_folder ../data_challenge/input/masks --submission_csv submission.csv

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
Heavily inspired from the Kaggle Carvana's third plate solution, it is a simple Keras sample allowing competitors to get started. Get started with `benchmark.ipynb`.
