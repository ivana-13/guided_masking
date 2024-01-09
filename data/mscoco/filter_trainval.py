import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import numpy as np
import tensorpack.dataflow as td
import json
import csv
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]
import sys
import pandas as pd
import zlib
import tqdm
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

corpus_path = "datasets/mscoco"
infile = os.path.join(corpus_path, 'tsv_files/trainval2014_obj36-36.tsv')
outfile = os.path.join(corpus_path, 'tsv_files/trainval2014_obj36-36_new.tsv')
captions = {}
df = json.load(open(os.path.join(corpus_path, 'annotations/train_ann.jsonl')))

for annotation in df["annotations"]:
    caption = annotation['sentences'][0]
    image_id = annotation['id']
    captions[image_id] = caption

length = 0
with open(outfile, 'w') as tsvfile:
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            image_id = item['img_id']
            img_id = int(item['img_id'].split('_')[-1])
            if img_id in captions.keys():
                writer.writerow(item)
                length += 1
            else:
                print("Not in the train set")
print("Length of the tsv file: %.0f" %(length))
