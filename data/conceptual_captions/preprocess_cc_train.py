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
import base64

maxInt = sys.maxsize
import time
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class Conceptual_Caption(td.RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        print('load')
        self.name = os.path.join(corpus_path, 'tsv_files/trainval2014_obj36-36_new.tsv')
        self.infiles = [self.name]
        self.counts = []

        self.captions = {}
        df = json.load(open(os.path.join(corpus_path, 'annotations/train_ann.jsonl')))

        for annotation in df["annotations"]:
            caption = annotation['sentences']
            image_id = annotation['id']
            self.captions[image_id] = caption
        

        with open(os.path.join(corpus_path, 'annotations/train_ann.jsonl')) as f:
            captions = json.load(f)["annotations"]
        self.num_caps = 5*len(captions)

    def __len__(self):
        print(self.num_caps)
        return self.num_caps

    def __iter__(self):
        with open(self.name) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader: 
                time.sleep(0.005)           
                image_id = item['img_id']
                img_id = int(item['img_id'].split('_')[-1])
                image_h = item['img_h']
                image_w = item['img_w']
                num_boxes = item['num_boxes']
                boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
                features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
                cls_prob = np.frombuffer(base64.b64decode(item['cls_prob']), dtype=np.float32).reshape(int(num_boxes), 1601)
                objects_id = np.frombuffer(base64.b64decode(item['objects_id']), dtype=np.int64)
                objects_conf = np.frombuffer(base64.b64decode(item['objects_conf']), dtype=np.float32)
                attrs_id = np.frombuffer(base64.b64decode(item['attrs_id']), dtype=np.int64)
                attrs_conf = np.frombuffer(base64.b64decode(item['attrs_conf']), dtype=np.float32)
                # cls_scores = np.frombuffer(base64.b64decode(item['classes']), dtype=np.float32).reshape(int(num_boxes), -1)
                attr_scores = np.frombuffer(base64.b64decode(item['attrs']), dtype=np.float32).reshape(int(num_boxes), -1)
                caption = self.captions[img_id]

                yield [features, cls_prob, objects_id, objects_conf, attrs_id, attrs_conf, attr_scores, boxes, num_boxes, image_h, image_w, image_id, caption]



corpus_path = sys.argv[1]
ds = Conceptual_Caption(corpus_path)
#ds1 = td.PrefetchDataZMQ(ds, 1)
ds1 = td.MultiProcessRunner(ds, 5000, 1)
td.LMDBSerializer.save(ds1, os.path.join(corpus_path, 'training_feat_all.lmdb'), write_frequency=5000)
