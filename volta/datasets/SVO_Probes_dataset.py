import json
from typing import Any, Dict, List
import random
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

from ._image_features_reader import ImageFeaturesH5Reader
from trankit import Pipeline

p = Pipeline(lang='english', gpu = True, cache_dir = './cache')
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def intersection(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
    )
    iw[iw < 0] = 0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih 

    return overlaps

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotations(annotations_jsonpath, negative_pairs_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""

    annotations_json = json.load(open(annotations_jsonpath))

    negative_pairs_json = json.load(open(negative_pairs_jsonpath))

    # Build an index which maps image id with a list of caption annotations.
    entries = []

    for pair in negative_pairs_json['pairs']:
        if pair[0] != pair[1]:
            if annotations_json[str(pair[0])][str(pair[1])][2] == []:
                annotations_json[str(pair[0])][str(pair[1])][2] = annotations_json[str(pair[0])][str(pair[1])][1].lower()
            entries.append(
            {   
                "caption": annotations_json[str(pair[0])][str(pair[1])][0].lower(),
                "foil": 1,
                "image_id": pair[1],
                "caption_number": pair[0],
                "caption_verb": annotations_json[str(pair[0])][str(pair[1])][1].lower(),
                "correct_verb": annotations_json[str(pair[0])][str(pair[1])][2].lower(),
                "caption_subject": annotations_json[str(pair[0])][str(pair[1])][3].lower(),
            }
            )
        else:
            second_annotation = list(annotations_json[str(pair[0])].keys())
            if len(second_annotation)>1:
                second_annotation = second_annotation[0]
            else:
                second_annotation = second_annotation[0]
            entries.append(
            {   
                "caption": annotations_json[str(pair[0])][second_annotation][0].lower(),
                "foil": 0,
                "image_id": pair[1],
                "caption_number": pair[0],
                "caption_verb": annotations_json[str(pair[0])][second_annotation][1].lower(),
                "correct_verb": annotations_json[str(pair[0])][second_annotation][1].lower(),
                "caption_subject": annotations_json[str(pair[0])][second_annotation][3].lower(),
            }
            )

    return entries

class SVO_ProbesClassificationDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        negative_pairs_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=20,
        max_region_num=101,
        num_locs=None,
        add_global_imgfeat=None,
        append_mask_sep=None, 
        vision_masking=False
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        
        self.task = task
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self._max_region_num = max_region_num
        self.num_labels = 2
        self.vision_masking = vision_masking
        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + ".pkl",
            )

        classes = ['__background__']
        with open(os.path.join(dataroot, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                classes.append(object.split(',')[0].lower().strip())
        self.object_vocab = classes
        if not os.path.exists(cache_path):
            self._entries = _load_annotations(annotations_jsonpath, negative_pairs_jsonpath)
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.
        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            # sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            # sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in sentence_tokens
            # ]

            #wo_punct_caption = entry["caption"].replace('.','')
            #wo_punct_caption = entry["caption"].replace(',','')


            lemmatizer = WordNetLemmatizer()
            if self.vision_masking == 'partial':
                if entry["foil"] == 0:
                    doc = entry['caption']
                    result = p.posdep(doc)['sentences'][0]['tokens']
                    subj = []
                    for i in range(len(result)):
                        if ("subj" in result[i]['deprel']) and (result[i]['upos']=='NOUN'):
                            if subj == []:
                                subj = result[i]['text']
                    if subj == []:
                        subj = entry['caption_subject']
                    if subj == []:
                        print(doc)
                entry['subj'] = subj
                    

            if self.task.split('_')[-1] == "masking":
                if entry["foil"] == 0:
                    doc = entry['caption']
                    result = p.posdep(doc)['sentences'][0]['tokens']
                    verb = []
                    subj = []
                    for i in range(len(result)):
                        if result[i]['upos'] == 'VERB':
                            verb = result[i]['text']
                        if ("subj" in result[i]['deprel']) and (result[i]['upos']=='NOUN'):
                            if subj == []:
                                subj = result[i]['text']
                    if subj == []:
                        subj = entry['caption_subject']
                    if subj == []:
                        print(doc)
                    if verb != []:
                        entry['caption_verb'] = str(verb)
                    else:
                        possible_verb = [word for word in (str(entry["caption"])).replace(',','').split() if lemmatizer.lemmatize(word, pos="v").find(entry["caption_verb"]) != -1]
                        if len(possible_verb)>=1:
                            entry['caption_verb'] = str(possible_verb[0])
                        else:
                            entry['caption_verb'] = ''
                    verb_tokens = self._tokenizer.encode(entry['caption_verb'])
                    masking = ' '
                    for i in range(len(verb_tokens)):
                        masking += '[MASK] '
                    if entry['caption'].split()[-1] == entry['caption_verb']:
                        split = entry['caption'].split()
                        entry['caption'] = ' '.join(split[:-1]) + masking
                    elif entry['caption'].split()[0] == entry['caption_verb']:
                        split = entry['caption'].split()
                        entry['caption'] =  masking + ' '.join(split[1:])
                    else:
                        if 'MASK' not in entry['caption']:
                            entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + ' ', masking, 1)
                        if 'MASK' not in entry['caption']:
                            entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + '.', masking, 1)
                        if 'MASK' not in entry['caption']:
                            entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + ',', masking, 1)
                    tokens = self._tokenizer.encode(entry["caption"])
                    if self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token) not in tokens:
                        print('Chyba: nemam zamaskovane nic v pozit')
                        print(entry['image_id'])
                        print(entry['caption'])
                        print(entry['caption_verb'])
                        print(verb_tokens)
                else:
                    doc = entry['caption']
                    result = p.posdep(doc)['sentences'][0]['tokens']
                    verb = []
                    for i in range(len(result)):
                        if result[i]['upos'] == 'VERB':
                            verb = result[i]['text']
                    if verb != []:
                        entry['caption_verb'] = str(verb)
                    else:
                        possible_verb = [word for word in (str(entry["caption"])).replace(',','').split() if lemmatizer.lemmatize(word, pos="v").find(entry["caption_verb"]) != -1]
                        if len(possible_verb)>=1:
                            entry['caption_verb'] = str(possible_verb[0])
                        else:
                            entry['caption_verb'] = ''
                    if entry['caption_verb'] != '':
                        print(entry['caption'])
                        if entry['caption_verb'][-3:] == 'ing':
                            entry["correct_verb"] = entry["correct_verb"] + 'ing'
                        elif entry['caption_verb'][-1] == 's':
                            entry["correct_verb"] = entry["correct_verb"] + 's'
                        elif entry['caption_verb'][-2:] == 'ed':
                            entry["correct_verb"] = entry["correct_verb"] + 'ed'
                        print(entry['correct_verb'])
                        if entry['caption'].split()[-1] == entry['caption_verb']:
                            split = entry['caption'].split()
                            entry['caption'] = ' '.join(split[:-1]) + ' ' + entry['correct_verb']
                        elif entry['caption'].split()[0] == entry['caption_verb']:
                            split = entry['caption'].split()
                            entry['caption'] =  entry['correct_verb'] + ' ' + ' '.join(split[1:])
                        else:
                            if entry['correct_verb'] not in entry['caption']:
                                entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + ' ',' ' + entry['correct_verb'] + ' ',1)
                            if entry['correct_verb'] not in entry['caption']:    
                                entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + '.',' ' + entry['correct_verb'] + '.',1)
                            if entry['correct_verb'] not in entry['caption']:
                                entry['caption'] = entry['caption'].replace(' ' + entry['caption_verb'] + ',',' ' + entry['correct_verb'] + ',',1)
                        print(entry['caption'])
                        verb_tokens = self._tokenizer.encode(entry['correct_verb'])
                        print(verb_tokens)
                        masking = ' '
                        for i in range(len(verb_tokens)):
                            masking += '[MASK] '
                        if entry['caption'].split()[-1] == entry['correct_verb']:
                            split = entry['caption'].split()
                            entry['caption'] = ' '.join(split[:-1]) + masking
                        elif entry['caption'].split()[0] == entry['correct_verb']:
                            split = entry['caption'].split()
                            entry['caption'] =  masking + ' '.join(split[1:])
                        else:
                            if 'MASK' not in entry['caption']:
                                entry['caption'] = entry['caption'].replace(' ' + entry['correct_verb'] + ' ', masking, 1)
                            if 'MASK' not in entry['caption']:
                                entry['caption'] = entry['caption'].replace(' ' + entry['correct_verb'] + '.', masking, 1)
                            if 'MASK' not in entry['caption']:
                                entry['caption'] = entry['caption'].replace(' ' + entry['correct_verb'] + ',', masking, 1)
                        print(entry['caption'])
                    else:
                        verb_tokens = []
                    tokens = self._tokenizer.encode(entry["caption"])
                    if self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token) not in tokens:
                        print('Chyba: nemam zamaskovane nic')
                        print(entry['image_id'])
                        print(entry['caption'])
                        print(entry['correct_verb'])
                        print(verb_tokens)
            else: 
                tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)


            if self.task.split('_')[-1] == "masking":
                output_label = []
                w = 0
                for i, token in enumerate(tokens):
                    if token == self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token):
                        output_label.append(verb_tokens[w])
                        w += 1
                    else:
                        output_label.append(-1)

            
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding
                if self.task.split('_')[-1] == "masking":
                    output_label = output_label + padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            if self.task.split('_')[-1] == "masking":
                entry['output_label'] = output_label
                entry['subj'] = subj

    def tensorize(self):

        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            if self.task.split('_')[-1] == "masking":
                output_label = torch.from_numpy(np.array(entry['output_label']))
                entry['output_label'] = output_label

    def __getitem__(self, index):

        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _, obj_labels = self._image_features_reader[image_id]
        image_mask = [1] * (int(num_boxes))
        boxes_reshaped = boxes.reshape((37,5))[1:37,:4]
        intersections = intersection(boxes_reshaped, boxes_reshaped)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        if self.vision_masking == 'partial':
            if entry['subj'] in list(self.object_vocab):
                obj_labels_backround = [i+1 for i in obj_labels]
                if list(self.object_vocab).index(entry['subj']) in obj_labels_backround:
                    min_label_subj = entry['subj']
                else: 
                    caption_subj = wordnet.synsets(entry['subj'])[0]
                    min_distance = 10000
                    min_label_subj = ''
                    for i in range(num_boxes-1):
                        try:         
                            word = self.object_vocab[(obj_labels[i]+1)].replace(' ', '_').split(',')[0]
                            label_subj = wordnet.synsets(word)[0]
                            if min_distance > caption_subj.shortest_path_distance(label_subj):
                                min_distance = caption_subj.shortest_path_distance(label_subj)
                                min_label_subj = self.object_vocab[(obj_labels[i]+1)]
                        except:
                            if min_label_subj == '':
                                print(image_id)
                                print(entry['subj'])
            else:
                try:
                    caption_subj = wordnet.synsets(entry['subj'])[0]
                except:
                    if entry['subj'] in ['i', 'you', 'he', 'she', 'it']:
                        entry['subj'] = 'person'
                    elif entry['subj'] in ['we', 'you', 'they']:
                        entry['subj'] = 'people'
                    elif entry['subj'] == 'emts':
                        entry['subj'] = 'paramedic'
                    elif entry['subj'] == 'midfielder':
                        entry['subj'] = 'player'
                    try:
                        caption_subj = wordnet.synsets(entry['subj'])[0]
                    except:
                        try:
                            caption_subj = wordnet.synsets(entry['caption_subject'])[0]
                        except:
                            print(image_id)
                            print(entry['subj'])
                            print(entry['caption'])
                min_distance = 10000
                min_label_subj = ''
                for i in range(num_boxes-1):
                    try:
                        word = self.object_vocab[(obj_labels[i]+1)].replace(' ', '_').split(',')[0]
                        label_subj = wordnet.synsets(word)[0]
                        if min_distance > caption_subj.shortest_path_distance(label_subj):
                            min_distance = caption_subj.shortest_path_distance(label_subj)
                            min_label_subj = self.object_vocab[(obj_labels[i]+1)]
                    except:
                        if min_label_subj == '':
                            print(image_id)
                            print(entry['subj'])
            list_of_subjects = list(self.object_vocab).index(min_label_subj)


            count = 0
            for i in range(num_boxes-1):
                if (int(obj_labels[i])+1) == list_of_subjects:
                    features[i] = 0
                    for j in np.arange(36): 
                        if intersections[i][j] > 0:
                            features[j] = 0
                            count += 1
                    break

            

        if self.vision_masking == 'complete':
            for j in np.arange(36): 
                features[j] = 0
                boxes[j] = 0
        
        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))

        caption = entry["token"]
        if self.task.split('_')[-1] == "masking":
            target = entry["output_label"]
        else:
            target = int(entry["foil"])
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        caption_number = torch.tensor(entry["caption_number"])

        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            image_id,
            obj_labels
        )

    def __len__(self):
        return len(self._entries)
