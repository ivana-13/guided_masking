# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
SVO = True
import sys
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np
from timeit import default_timer as timer

import torch
import torch.distributed as dist
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.tokenization_bert import BertTokenizer

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

from volta.config import BertConfig
from volta.optimization import RAdam
from volta.encoders import BertForVLTasks
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume
from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Training
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Scheduler
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")
    parser.add_argument("--task_specific_tokens", action="store_true",
                        help="whether to use task specific tokens for the multi-task learning.",)
    parser.add_argument("--evaluation", default="best_5", 
                        help="wheter to use the ViLBERTscore for evaluation, comparison with BERT, or just the best 5 words")
    parser.add_argument("--vision_masking", default=False,
                        help="wheter to mask bounding box with the subject or not")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    base_lr = task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timestamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + prefix)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val = LoadDataset(args, config, task_cfg, args.task)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.task_specific_tokens:
        config.task_specific_tokens = True
    else:
        config.task_specific_tokens = False

    # Model
    model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print('Number of parameters in the model: %.0f', pp)

    zero_shot = True
    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)

    # Optimization details
    freeze_layers(model)
    criterion = LoadLoss(task_cfg, args.task)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
    #if default_gpu:
        #print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=base_lr,
                          eps=args.adam_epsilon,
                          betas=args.adam_betas,
                          correct_bias=args.adam_correct_bias)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    num_train_optim_steps = (task2num_iters[task] * args.num_train_epochs // args.grad_acc_steps)
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, max_score = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save starting model
    #if start_epoch == 0:
        #save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)
    
    if zero_shot == False:
        # Print summary
        if default_gpu:
            summary_parameters(model, logger)
            print("***** Running training *****")
            print("  Num Iters: ", task2num_iters[task])
            print("  Batch size: ", batch_size)
            print("  Num steps: %d" % num_train_optim_steps)

        # Train
        for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(dl_train):
                iter_id = start_iter_id + step + (epoch_id * len(dl_train))

                loss, score = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion)

                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                loss.backward()

                if (step + 1) % args.grad_acc_steps == 0:
                    # Clip gradient
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                    optimizer.step()

                    if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                        scheduler.step()

                    model.zero_grad()
                    global_step += 1

                    if default_gpu:
                        tb_logger.step_train(epoch_id, iter_id, float(loss), float(score),
                                            optimizer.param_groups[0]["lr"], task, "train")

                if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                    tb_logger.showLossTrain()

                # Decide whether to evaluate task
                if iter_id != 0 and iter_id % task2num_iters[task] == 0:
                    score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger, args)
                    if score > max_score:
                        max_score = score
                        #save(save_path, logger, epoch_id, model, optimizer, scheduler,
                        #    global_step, tb_logger, default_gpu, max_score, is_best=True)

            #save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)

        tb_logger.txt_close()
    else:
        epoch_id = 0
        start = timer()
        print("Zero-shot learning, starting evaluation")
        evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger, args)
        end = timer()
        print("Time:")
        print(end - start)

def evaluate(config, dataloader_val, task_cfg, device, task_id, model, criterion, epoch_id, default_gpu, tb_logger, args):
    model.eval()
    score_avg_temp = 0
    score_posit_temp = 0
    score_negat_temp = 0
    data_size_temp = 0
    posit_samples_temp = 0
    negat_samples_temp = 0
    samples_temp = 0
    positive_pair_good_classification = []
    positive_pair_bad_classification = []
    negative_pair_good_classification = []
    negative_pair_good_classification_img = []
    negative_pair_bad_classification = []
    negative_pair_bad_classification_img = []
    dictionary_of_representations = {}

    classes = ['__background__']
    with open('objects_vocab.txt') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    
    index_cls = 0
    for cls in classes:
        if cls == 'man':
            break
        else:
            index_cls += 1
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
    masked_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    probs_4place = 0
    probs_5place = 0
    for i, batch in enumerate(dataloader_val):
        if SVO == True:
            if task_cfg[task_id]["type"] == "VL-binary-classifier":
                features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id, obj_labels = batch
                batch = features, spatials, image_mask, caption, target, input_mask, segment_ids, obj_labels
                loss, score, score_posit, score_negat, batch_size, posit_samples, negat_samples, positive_pair_good_classification, \
                positive_pair_bad_classification, negative_pair_good_classification,  negative_pair_good_classification_img, \
                negative_pair_bad_classification, negative_pair_bad_classification_img, dictionary_of_representations = \
                ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion, tokenizer, positive_pair_good_classification,\
                positive_pair_bad_classification, negative_pair_good_classification, negative_pair_good_classification_img,\
                negative_pair_bad_classification, negative_pair_bad_classification_img, dictionary_of_representations, index_cls, args.evaluation)
            elif task_cfg[task_id]["type"] == "word-prediction":
                features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id, obj_labels = batch
                batch = features, spatials, image_mask, caption, target, input_mask, segment_ids, obj_labels
                loss, score, score_posit, score_negat, batch_size, samples, x, y = \
                ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion, tokenizer, positive_pair_good_classification,\
                positive_pair_bad_classification, negative_pair_good_classification, negative_pair_good_classification_img,\
                negative_pair_bad_classification, negative_pair_bad_classification_img, dictionary_of_representations, index_cls, args.evaluation)
                probs_4place += x
                probs_5place += y
            elif task_cfg[task_id]["type"] == "token-prediction":
                features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id, obj_labels = batch
                batch = features, spatials, image_mask, caption, target, input_mask, segment_ids, obj_labels
                loss, score, score_posit, score_negat, batch_size, samples, x, y = \
                ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion, tokenizer, positive_pair_good_classification,\
                positive_pair_bad_classification, negative_pair_good_classification, negative_pair_good_classification_img,\
                negative_pair_bad_classification, negative_pair_bad_classification_img, dictionary_of_representations, index_cls, args.evaluation)
            else:
                vil_prediction = ForwardModelsVal(config, task_cfg, \
                device, task_id, batch, model, criterion, positive_pair_good_classification, positive_pair_bad_classification, negative_pair_good_classification,\
                negative_pair_good_classification_img, negative_pair_bad_classification, negative_pair_bad_classification_img)
        else:
            features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id = batch
            loss, score, score_posit, score_negat, batch_size, posit_samples, negat_samples, positive_pair_good_classification, \
            positive_pair_bad_classification, negative_pair_good_classification, negative_pair_bad_classification = ForwardModelsVal(config, task_cfg, \
            device, task_id, batch, model, criterion, positive_pair_good_classification, positive_pair_bad_classification, negative_pair_good_classification,\
            negative_pair_bad_classification)

        
        if task_cfg[task_id]["type"] == "word-prediction":
            score_avg_temp += score
            score_posit_temp += score_posit
            score_negat_temp += score_negat
            data_size_temp += batch_size
            samples_temp += samples
        else:
            score_avg_temp += score
            score_posit_temp += score_posit
            score_negat_temp += score_negat
            data_size_temp += batch_size
            posit_samples_temp += posit_samples
            negat_samples_temp += negat_samples


        tb_logger.step_val(epoch_id, float(loss), float(score), task_id, batch_size, "val")
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()
        if int(100*i/len(dataloader_val))%5 == 0:
            print ((score_posit_temp * 100.0)/data_size_temp)

    if task_cfg[task_id]["type"] == "word-prediction":
        print("Cumulative results: score %.3f score_posit %.0f score_negat %.0f samples %.0f" % (
                (score_avg_temp * 100.0)/data_size_temp,
                score_posit_temp,
                score_negat_temp,
                samples_temp,
                #posit_samples_temp,
                #negat_samples_temp
                ))
        print(probs_4place/samples_temp)
        print(probs_5place/samples_temp)
    else:
        print("Cumulative results: score %.3f score_posit %.0f score_negat %.0f posit_samples %.0f negat_samples %.0f" % (
                (score_avg_temp * 100.0)/data_size_temp,
                score_posit_temp,
                score_negat_temp,
                posit_samples_temp,
                negat_samples_temp
                ))
    if SVO == True:
        dict = {'data': positive_pair_good_classification}
        with open('positive_pair_good_classification.json', 'w') as fp:
            json.dump(dict, fp)
        dict = {'data': positive_pair_bad_classification}
        with open('positive_pair_bad_classification.json', 'w') as fp:
            json.dump(dict, fp)
        dict = {'data': negative_pair_good_classification}
        with open('negative_pair_good_classification.json', 'w') as fp:
            json.dump(dict, fp)
        dict = {'data': negative_pair_good_classification_img}
        with open('negative_pair_good_classification_img.json', 'w') as fp:
            json.dump(dict, fp)
        dict = {'data': negative_pair_bad_classification}
        with open('negative_pair_bad_classification.json', 'w') as fp:
            json.dump(dict, fp)
        dict = {'data': negative_pair_bad_classification_img}
        with open('negative_pair_bad_classification_img.json', 'w') as fp:
            json.dump(dict, fp)
        print(type(dictionary_of_representations))
        with open('representations.json', 'w') as fp:
            json.dump(dictionary_of_representations, fp)
    
    #score = tb_logger.showLossVal(task_id)
    #model.train()


if __name__ == "__main__":
    main()
