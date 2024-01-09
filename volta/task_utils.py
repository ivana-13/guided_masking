# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForMaskedLM, AutoTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer

from volta.datasets import DatasetMapTrain, DatasetMapEval
from nltk.stem import WordNetLemmatizer
from volta.datasets._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.vocab.items():
        if index == integer:
            return word
    return None

def bert_score(ref_embedding, hyp_embedding, ref_masks, hyp_masks, use_idf=False):
    batch_size = ref_embedding.size(0)

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim[:,1:-1, 1:-1]
    
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks[:,1:-1,1:-1] # default
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim*masks
    
    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_masks = hyp_masks[:, 1:-1].float()
    ref_masks = ref_masks[:, 1:-1].float()

    hyp_masks.div_(hyp_masks.sum(dim=1, keepdim=True))
    ref_masks.div_(ref_masks.sum(dim=1, keepdim=True))
    
    precision_scale = hyp_masks.to(word_precision.device).float()
    recall_scale = ref_masks.to(word_recall.device).float()
    
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)
    return P.detach().cpu().numpy(), R.detach().cpu().numpy(), F.detach().cpu().numpy() 

def ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion, tokenizer, positive_pair_good_classification, \
positive_pair_bad_classification, negative_pair_good_classification, negative_pair_good_classification_img, negative_pair_bad_classification, \
negative_pair_bad_classification_img, dictionary_of_representations, index_cls, vilbertscore_evaluation):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    SVO = True

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        if SVO == True:
            features, spatials, image_mask, question, target, input_mask, segment_ids, obj_labels = batch
        else:
            features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        raise NotImplementedError("dialog process for validation")

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    
    vil_prediction, vision_prediction, linguisic_prediction, _, prediction_scores_t = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask, task_tokens)

    # for i in range(len(question_id)):
    #     if index_cls in obj_labels[i]:
    #         boxes_index = [j for j in range(len(obj_labels[i])) if obj_labels[i][j]==index_cls]
    #         new_dict = {}
    #         new_dict['caption id'] = int(caption_number[i])
    #         new_dict['text representation'] = linguisic_prediction[i].cpu().detach().tolist()
    #         features_before = []
    #         features_after = []
    #         for j in boxes_index:
    #             features_before.append(features[i].cpu().detach().tolist()[j])
    #             features_after.append(vision_prediction[i].cpu().detach().tolist()[j])
    #         new_dict['image representation before'] = features_before
    #         new_dict['image representation after'] = features_after
    #         if str(int(question_id[i])) in dictionary_of_representations.keys():
    #             list_of_dicts = dictionary_of_representations[str(int(question_id[i]))]
    #             del dictionary_of_representations[str(int(question_id[i]))]
    #             list_of_dicts += [new_dict]
    #             dictionary_of_representations[str(int(question_id[i]))] = list_of_dicts
    #         else:
    #             dictionary_of_representations[str(int(question_id[i]))] = [new_dict]

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit":
        # loss = criterion(vil_prediction, target)
        # loss = loss.mean() * target.size(1)
        # _, select_idx = torch.max(vil_prediction, dim=1)
        # select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        # batch_score = torch.sum(select_target > 0.5).item()
        loss = 0

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        sftmx = nn.Softmax(dim=1)
        vil_binary_prediction_probs = sftmx(vil_prediction)
        #print(target)
        score_avg, score_posit, score_negat = compute_score_with_logits(vil_prediction, target)
        batch_score = score_avg
        negat_samples = target.sum()
        posit_samples = (batch_size - negat_samples)
        # if SVO == True:
        #     positive_pair = question_id[target==0]
        #     positive_pair_good_classification += (caption_number[((target==0) & (torch.max(vil_prediction, 1)[1].data==0))]).tolist()
        #     positive_pair_bad_classification += (caption_number[((target==0) & (torch.max(vil_prediction, 1)[1].data==1))]).tolist()
        #     negative_pair_good_classification += (caption_number[((target==1) & (torch.max(vil_prediction, 1)[1].data==1))]).tolist()
        #     negative_pair_good_classification_img += (question_id[((target==1) & (torch.max(vil_prediction, 1)[1].data==1))]).tolist()
        #     negative_pair_bad_classification += (caption_number[((target==1) & (torch.max(vil_prediction, 1)[1].data==0))]).tolist()
        #     negative_pair_bad_classification_img += (question_id[((target==1) & (torch.max(vil_prediction, 1)[1].data==0))]).tolist()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(prediction_scores_t, target.view(-1))
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()
    elif task_cfg[task_id]["type"] == "token-prediction":    
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(prediction_scores_t, target.view(-1))
        loss = loss.mean()
        sftmx = nn.Softmax(dim=1)
        for i in range(batch_size):
            masked_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            mask_indices = question[i] == masked_token_id
            evaluated_caption = prediction_scores_t[(i*20):((i+1)*20),:]
            evaluated_verb = sftmx(evaluated_caption[target[i] > 0,:])
            sorted, indices = torch.sort(evaluated_verb[0,:])
            original = word_for_id(target[i][target[i]!=-1][0].data, tokenizer)
            

    elif task_cfg[task_id]["type"] == "word-prediction":
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(prediction_scores_t, target.view(-1))
        loss = loss.mean()
        sftmx = nn.Softmax(dim=1)
        score_negat = 0
        score_posit = 0
        score_neutral = 0
        if (vilbertscore_evaluation == 'compare_with_BERT') or (vilbertscore_evaluation == 'BERT'):
            model_bert = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device)
            probs = model_bert(question).logits.softmax(dim=-1)
        probs_4place = 0
        probs_5place = 0
        for i in range(batch_size):
            masked_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            mask_indices = question[i] == masked_token_id
            score_posit_idx = 0
            score_negat_idx = 0
            score_neutral_idx = 0
            evaluated_caption = prediction_scores_t[(i*20):((i+1)*20),:]
            evaluated_verb = sftmx(evaluated_caption[target[i] > 0,:])
            if (vilbertscore_evaluation == 'compare_with_BERT') or (vilbertscore_evaluation == 'BERT'):
                evaluated_caption_bert = probs[i]
                evaluated_verb_bert = evaluated_caption_bert[mask_indices,:]
            for j in range(evaluated_verb.size()[0]):
                sorted, indices = torch.sort(evaluated_verb[j,:])
                if (vilbertscore_evaluation == 'compare_with_BERT') or (vilbertscore_evaluation == 'BERT'):
                    sorted_bert, indices_bert = torch.sort(evaluated_verb_bert[j,:])
                list_of_verbs = []
                list_of_verbs_not_lemmatized  = []
                list_of_verbs_bert = []
                list_of_verbs_vilbert = []
                lemmatizer = WordNetLemmatizer()
                original = lemmatizer.lemmatize(word_for_id(target[i][target[i]!=-1][j].data, tokenizer), pos ="v")
                max_prob = 0
                max_prob_bert = 0
                list_of_probs = []
                list_of_probs_bert = []
                f5 = []
                for k in range(5):
                    list_of_verbs += [lemmatizer.lemmatize(''.join(filter(str.isalnum, word_for_id(indices[-(k+1)].data, tokenizer))), pos = "v")]
                    list_of_verbs_not_lemmatized += [''.join(filter(str.isalnum, word_for_id(indices[-(k+1)].data, tokenizer)))]
                    if vilbertscore_evaluation == 'vilbertscore':
                        list_of_verbs_vilbert += [''.join(filter(str.isalnum, word_for_id(indices[-(k+1)].data, tokenizer)))]
                    if k ==3:
                        probs_4place += sorted[-(k+1)].data
                    if k == 4:
                        probs_5place += sorted[-(k+1)].data
                    list_of_probs += [sorted[-(k+1)].data]
                    if vilbertscore_evaluation == "vilbertscore":
                        reference = torch.add(torch.mul(torch.add(mask_indices,-1),-1)*question[i], mask_indices*target[i]).unsqueeze(0)
                        if evaluated_verb.size()[0] == 1:
                            candidate = torch.add(torch.mul(torch.add(mask_indices,-1),-1)*question[i], mask_indices*indices[-(k+1)].data).unsqueeze(0)
                        else:
                            predicted_word = -1
                            predicted_candidate = mask_indices.clone().int()
                            for p in range(evaluated_verb.size()[0]):
                                if j == p:
                                    predicted_word = indices[-(k+1)].data
                                else:
                                    sorted_help, indices_help = torch.sort(evaluated_verb[p,:])
                                    predicted_word = indices_help[-1].data
                                for m in range(predicted_candidate.shape[0]):
                                    if predicted_candidate[m] == 1:  
                                        predicted_candidate[m] = predicted_word
                                        break
                            candidate = torch.add(torch.mul(torch.add(mask_indices,-1),-1)*question[i],predicted_candidate).unsqueeze(0)
                        with torch.no_grad():
                            st_c, sv_c, pt_c, pv_c, att_c= model.bert(reference, features[i,:,:].unsqueeze(0), spatials[i,:,:].unsqueeze(0), \
                            segment_ids[i,:].unsqueeze(0), input_mask[i,:].unsqueeze(0), image_mask[i,:].unsqueeze(0), task_tokens[i,:].unsqueeze(0))
                        with torch.no_grad():
                            st_g, sv_g, pt_g, pv_g, att_g = model.bert(candidate, features[i,:,:].unsqueeze(0), spatials[i,:,:].unsqueeze(0), \
                            segment_ids[i,:].unsqueeze(0), input_mask[i,:].unsqueeze(0), image_mask[i,:].unsqueeze(0), task_tokens[i,:].unsqueeze(0))
                        p, r, f = bert_score(st_c[-1].unsqueeze(0), st_g[-1].unsqueeze(0), input_mask[i ,:].unsqueeze(0), input_mask[i, :].unsqueeze(0))
                        del st_c, sv_c, pt_c, pv_c, att_c
                        del st_g, sv_g, pt_g, pv_g, att_g
                        del p, r
                        f5.append(f)        
                    if str(original) == str(lemmatizer.lemmatize(''.join(filter(str.isalnum, word_for_id(indices[-(k+1)].data, tokenizer))), pos = "v")):
                        if max_prob < sorted[-(k+1)].data:
                            max_prob = sorted[-(k+1)].data
                    if vilbertscore_evaluation == "compare_with_BERT":
                        list_of_verbs_bert += [lemmatizer.lemmatize(word_for_id(indices_bert[-(k+1)].data, tokenizer), pos = "v")]
                        list_of_probs_bert += [sorted_bert[-(k+1)].data]
                        if str(original) == str(lemmatizer.lemmatize(word_for_id(indices_bert[-(k+1)].data, tokenizer), pos = "v")):
                            if max_prob_bert < sorted_bert[-(k+1)].data:
                                max_prob_bert = sorted_bert[-(k+1)].data
                    if vilbertscore_evaluation == 'BERT':
                        list_of_verbs_bert += [lemmatizer.lemmatize(word_for_id(indices_bert[-(k+1)].data, tokenizer), pos = "v")]
                        list_of_probs_bert += [sorted_bert[-(k+1)].data]
                #print(original)
                if vilbertscore_evaluation == "best_5":
                    if str(original) in list_of_verbs:
                        score_posit_idx += 1
                    else:
                        score_negat_idx += 1
                if vilbertscore_evaluation == "BERT":
                    if str(original) in list_of_verbs_bert:
                        score_posit_idx += 1
                    else:
                        score_negat_idx += 1
                if vilbertscore_evaluation == "compare_with_BERT":
                    if max_prob == 0:
                        max_prob = evaluated_verb[j,target[i][target[i]!=-1][j].data]
                    if max_prob_bert == 0:
                        max_prob_bert = evaluated_verb_bert[j,target[i][target[i]!=-1][j].data]
                    if max_prob_bert < max_prob:
                        score_posit_idx += 1
                    else:
                        score_negat_idx += 1
            if vilbertscore_evaluation == "vilbertscore":
                if evaluated_verb.size()[0] != 0:
                    f5a = np.average(f5, axis=0) 
                    score_posit += float(f5a)
                else:
                    score_neutral_idx = 1
                listing = [word_for_id(words, tokenizer) for words in question[i]]
                # print(' '.join(listing))
                # print(list_of_verbs_vilbert)
                # print(word_for_id(target[i][target[i]!=-1][j].data, tokenizer))
                # print(f5)
                # print(float(f5a))
            if vilbertscore_evaluation in ['compare_with_BERT', 'BERT', 'best_5']:
                if evaluated_verb.size()[0] == 0:
                    verb_indexes = 1
                    score_neutral_idx = 1
                else:
                    verb_indexes = evaluated_verb.size()[0]
                if score_neutral_idx != 1:
                    if score_posit_idx/verb_indexes >= 0.5:
                        score_posit += 1
                    else:
                        score_negat += 1
                else:
                    score_neutral += 1
        batch_score = score_posit/target.size()[0]
        if vilbertscore_evaluation == 'vilbertscore':
            samples = target.size()[0] - score_neutral_idx
        else:
            samples = target.size()[0]
    if task_cfg[task_id]["type"] == "VL-binary-classifier":
        return float(loss), float(batch_score), float(score_posit), float(score_negat), batch_size, float(posit_samples), float(negat_samples), \
        positive_pair_good_classification, positive_pair_bad_classification, negative_pair_good_classification, negative_pair_good_classification_img,\
        negative_pair_bad_classification, negative_pair_bad_classification_img, dictionary_of_representations
    elif task_cfg[task_id]["type"] == "word-prediction":
        return float(loss), float(batch_score), float(score_posit), float(score_negat), batch_size, samples, probs_4place, probs_5place
    else:
        return vil_prediction


def ForwardModelsTrain(config, task_cfg, device, task_id, batch, model, criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
        segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask)

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    return loss, batch_score


def LoadLoss(task_cfg, task_id):
    task = "TASK" + task_id
    loss = LossMap[task_cfg[task]["loss"]]
    return loss


def LoadDataset(args, config, task_cfg, task_id, split="trainval"):
    #tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task]["batch_size"] // args.grad_acc_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    dset_train, dset_train, task2num_iters = None, None, {}
    if "train" in split:
        dset_train = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            negative_pairs_jsonpath=task_cfg[task]["negative_pairs_jasonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            vision_masking = args.vision_masking
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(dset_train)
        else:
            train_sampler = DistributedSampler(dset_train)
        dl_train = DataLoader(
            dset_train,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=args.drop_last,
        )
        task2num_iters = {task: len(dl_train)}

    dset_val, dl_val = None, None
    if "val" in split:
        dset_val = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            negative_pairs_jsonpath=task_cfg[task]["negative_pairs_jasonpath"],
            split=task_cfg[task]["val_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            vision_masking=args.vision_masking
        )
        dl_val = DataLoader(
            dset_val,
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=args.drop_last,
        )

    return batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val


def LoadDatasetEval(args, config, task_cfg, task_id):
    #tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    if task_name.startswith("Retrieval"):
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            num_subiters=args.num_subiters,
        )
    else:
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            vision_masking=args.vision_masking
        )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argma
    scores_negat = (logits * labels).sum()
    scores_posit = ((logits-1)*(labels-1)).sum()
    scores_avg = (scores_posit + scores_negat).sum()
    return scores_avg, scores_posit, scores_negat


def EvaluatingModel(config, task_cfg, device, task_id, batch, model, dataloader, criterion, results, others):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    with torch.no_grad():
        vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                           segment_ids, input_mask, image_mask)

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others
