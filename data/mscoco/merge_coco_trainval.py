import os
import sys
import argparse
import csv
from tqdm import tqdm
import time
time.sleep(600)

# SPLIT to its folder name under IMG_ROOT
SPLIT2DIR = {
    'train': 'train2014',
    'valid': 'val2014',
    'test': 'test2015',
    'trainval': 'trainval2014'
}


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def merge_tsvs(fname, total_group):
    fnames = ['train2014_obj36-36.tsv', 'val2014_obj36-36.tsv']
    print(fnames)
    outfile = fname
    with open(outfile, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        found_ids = set()
        for infile in fnames:
            print(infile)
            dir_infile = os.path.join(args.datadir, infile )
            with open(dir_infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in tqdm(reader):
                    img_id = item['img_id']
                    if img_id not in found_ids:
                        writer.writerow(item)
                        found_ids.add(img_id)
                    else:
                        print("Chyba")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--datadir', type=str, default='data/mscoco')
    parser.add_argument('--total_group', type=int, default=1,
                        help="the number of group for extracting")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    args.outfile = os.path.join(args.datadir, "trainval2014_obj36-36.tsv" )
    
    # Generate TSV files, normally do not need to modify
    merge_tsvs(args.outfile, args.total_group)
