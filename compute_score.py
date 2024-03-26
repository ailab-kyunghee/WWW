import argparse
import pickle
import json
import os
import numpy as np
from tqdm import tqdm

import torch
import pandas as pd

from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

import clip
import utils.metric_utils as metric

def main(args):
    print(args)
    print('* Start Metric: ', args.target_pkl.split('/')[-1])
    
    target_save_dir = metric.make_dir(args.target_pkl, args.mode, args.save_dir, args.c_dissect)
    metric_score = []
    metric_dir = os.path.join(target_save_dir, 'score.json')
    gt, our_labels = metric.load_files(args.gt, args.target_pkl, args.tem, args.mode, args.c_dissect)

    ## accuracy
    if 'acc' in args.metric:
        ours_acc, _ = metric.ours_accuracy(our_labels, gt)
        acc_dict = {'Accuracy':ours_acc}
        metric_score.append(acc_dict)
        print('Finish : Accuracy')

    ## similarity
    if 'sim' in args.metric:
        mpnet_model = SentenceTransformer('all-mpnet-base-v2')
        clip_model, _ = clip.load(args.clip_name, device=args.device)

        similarity = metric.similarities(clip_model, mpnet_model, our_labels, gt, args.batch_size, args.device)
        metric_score.append(similarity)
        print('Finish : Cosine Similarity')

    if 'pr' in args.metric:
        pr = metric.pr(our_labels, gt)
        metric_score.append(pr)
        print('Finish : Precision & Recall')

    metric_score[0], metric_score[1], metric_score[2] = metric_score[1], metric_score[2], metric_score[0]
    with open(metric_dir,'w') as f:
        json.dump(metric_score, f, indent=4)
    print('score :', metric_score)
    print('end')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Metric')
    parser.add_argument('--gt', type=str, default='/data/imagenet_labels.txt')
    parser.add_argument('--target_pkl', type=str, default='/resnet50_imagenet_fc.pkl')
    parser.add_argument('--clip_name', type=str, default='ViT-B/16')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='results')    
    parser.add_argument('--metric', type=str, default='acc, sim, pr', help='acc,sim,pr')
    parser.add_argument('--c_dissect', type=int, default=1)
    parser.add_argument('--is_debug', type=str, default=False)
    parser.add_argument('--mode', type=str, default='www', help='www, cd')

    args = parser.parse_args()

    args.metric = args.metric.replace(' ','').split(',')
    if 'tem' in args.target_pkl.split('/')[-1].split('_'):
        args.tem = 'yes'
    else :
        args.tem = 'no'

    main(args)