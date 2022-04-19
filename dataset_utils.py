
import argparse
import os
from datasets import ClassLabel, Value, load_dataset, Features
from typing import AnyStr
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

def prompt_preparing(text, template): #to be fix, a tiny demo
    temp1, temp2 = template.split('<sent>')
    return temp1 + text + temp2

def make_dataset(config, tokenizer, dataset):
    task_name = config.task_name.lower()
    label_ids = {}
    # to be fix
    label_ids = config.label_ids
    #dataset_column = ['review', 'sentiment']
    max_len = config.max_len
    
    example_no = 0
    #example by imdb
    def data_preprocess(examples):
        total = len(examples['sentiment'])
        sent_feature = []
        for i, text in enumerate(examples['review']):
            review_sent = prompt_preparing(text, config.template)
            sent_feature.append(tokenizer(review_sent, truncation=True, max_length=max_len, padding='max_length', )) 

        feature = {"input_ids":[], "attention_mask":[], "labels":[], "idx":[], }
        for i, label in enumerate(examples['sentiment']):
            feature["input_ids"].append(sent_feature[i]["input_ids"])
            feature["attention_mask"].append(sent_feature[i]["attention_mask"])
            feature["labels"].append(label_ids[label])     
            feature['idx'].append(i)     
        return feature
    
    ret_dataset = dataset.map(data_preprocess, batched=True, remove_columns=dataset.column_names, )  #version [remove_columns]
    ret_dataset.set_format(type='torch',)
    
    #ret_dataset = dataset.map(data_preprocess, batched=True, input_columns=dataset_column) 
    
    if config.option == "train" and config.use_cl_exmp:
        train_emb = np.load(os.path.join(config.data_dir, "k-shot", config.task_name, "{}-{}".format(config.num_k, config.seed), "{}_sbert-{}.npy".format(config.option, config.sbert_model))) #(k*num_label, L_h) #to be fix

        assert(len(train_emb) == len(dataset))

        sim_score = util.pytorch_cos_sim(train_emb, train_emb)
        #not to computer Similarity with itself
        eyes = torch.eye(sim_score.shape[0])
        sim_score *= 1-eyes

        label_sim_score = [] #label * (idx, k*top_rate)
        for i in range(len(label_ids)):
            label_sim_score.append(sim_score[:, i*config.num_k:(i+1)*config.num_k].topk(int(config.top_rate * config.num_k), dim=-1).indices) #append (idx, k*top rate)
        #import pdb; pdb.set_trace()
        exmp_sim_score = torch.stack(label_sim_score, dim=1) #(idx,label, k*top rate)
        return ret_dataset, exmp_sim_score
    
    return ret_dataset
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, nargs='+', default='imdb')
    parser.add_argument("--seed", type=int, nargs="+", default=42, help="Seeds for data splits")
    parser.add_argument('--model_name_or_path', type=str, default='../plm/roberta-base')
    parser.add_argument('--dataset', type=str, default='data/k-shot/IMDB/16-42/train.csv')
    parser.add_argument('--template', type=str, default='the movie is <mask>, <sent>')
    parser.add_argument('--option', type=str, default='train', choices=['train', 'dev'])
    parser.add_argument('--sbert_model', type=str, default='roberta-large')
    parser.add_argument('--top_rate', type=float, default=0.5)
    parser.add_argument('--num_k', type=int, default=16)
    parser.add_argument('--data_dir',type=str, default='data/k-shot/IMDB/16-42')

    config = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False)
    file_type = config.dataset.split('.')[-1]
    exmp_dataset = load_dataset(file_type, name='imdb_fs', data_files=config.dataset, cache_dir='dataset_cache', )
    output = make_dataset(config, tokenizer, exmp_dataset['train'])
    import pdb; pdb.set_trace()