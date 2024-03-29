import argparse
from email.policy import default
import logging
import os
from typing import List
from datasets import load_dataset

import numpy as np
import torch

from models import TransformerModelWrapper
from task_utils import set_config_for_task, set_seed
from dataset_utils import make_dataset
from trainer import train_process, test_process

logger = logging.getLogger('main')
parser = argparse.ArgumentParser(
    description="command line and hyperparameters for prmt-tuning, \
                we devide into three type of args: \
                training args, modeling args, running args")

#running args
parser.add_argument("--task_name", type=str, default="ag_news", help="task name")
#to be fix, maybe not useful
parser.add_argument("--cuda_list", type=int, default=[0,1], nargs='+',  help="list all cuda for training")
parser.add_argument("--main_cuda", type=str, default='cuda:0')
parser.add_argument("--metrics", type=List[str], help="metrics for task, best model is decided by the 1st metrics")
parser.add_argument("--output_dir", type=str, default="model")
parser.add_argument("--cache_dir", type=str, default="cache")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--model_name_or_path", type=str, default="../plm/roberta-large", help="name or path of pretrained model")
parser.add_argument("--save_result", type=bool, default=True, help="whether save results when eval/test ")
parser.add_argument("--num_label", type=int,  help="number of labels, decide by task")
parser.add_argument("--label_ids", type=dict, help="labels map into ids, decide by task")
parser.add_argument("--label_words", type=list, help="label verbalizer, decide by task")
parser.add_argument("--dataset_column", type=str,)
parser.add_argument("--max_len", type=int, help="max length of input tokens, decide by task")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--template", type=str,  help="prompt template, decide by task")
parser.add_argument("--option", type=str, default="train", choices=["train", "test"], help="running train or test")
parser.add_argument("--do_predict", action="store_true", help="whether to test on full dataset when evaluating")
parser.add_argument("--sbert_model", type=str, default="roberta-large", help="pretrained model for sbert")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--async_test", action="store_true")
parser.add_argument("--async_cuda", type=str, default="cuda:1", help="cuda for test asynchronously, should be different from main_cuda")
parser.add_argument("--async_batch_size", type=int, default=16, help="batch size for test asynchronously")

#training args
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_stage", type=int, default=1, help="training stage")
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--num_train_epochs", type=int, default=400, help="total training epochs")
parser.add_argument("--grad_accu", type=int, default=2, help="gradient accumulate steps")
parser.add_argument("--warmup_rate", type=float, default=0.0, help="warmup step percent")
parser.add_argument("--num_logging_epochs", type=int, default=10, help="log training state every X epochs")
parser.add_argument("--num_eval_epochs", type=int, default=40, help="evaluate every X epochs when training")
parser.add_argument("--ce_loss", type=bool, default=True, help="whether to use ce_loss")
parser.add_argument("--alpha", type=float, default=0.01, help="weight for MLM loss when combining losses")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping")
parser.add_argument("--use_cl_exmp", action='store_true', default=True, help="whether to use contrastive examples")
parser.add_argument("--cl_temp", type=float, default=1, help="temperature for contrastive learning")
parser.add_argument("--num_k", type=int, default=16, help="k-shot")
parser.add_argument("--top_rate", type=float, default=0.5, help="similarity rank for contrastive examples")
parser.add_argument("--num_label_words", type=int, default=5, help="number of label words each class for contrastive learning")

#modeling args
parser.add_argument("--weight_decay", type=float, default=0.01, )
parser.add_argument("--learning_rate", type=float, default=1e-5, help="lr for all param, except for embedding in some case")
parser.add_argument("--embedding_learning_rate", type=float, default=5e-5, help="lr for embedding")
parser.add_argument("--fix_other_embeddings",action='store_true', help="fix embedding which is not for soft labels")
parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")

#parser.add_argument("--", type=, default=, help=)


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    config = parser.parse_args()
    set_seed(config.seed)
    set_config_for_task(config)
    model = TransformerModelWrapper(config)
    if config.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    

    data_dir = os.path.join(config.data_dir, "k-shot", config.task_name, "{}-{}".format(config.num_k, config.seed, ))
    train_dataset = load_dataset('csv', name=config.task_name, data_files=data_dir+"/train.csv", cache_dir=config.cache_dir)
    eval_dataset = load_dataset(path='csv', name=config.task_name, data_files={'dev':data_dir+"/dev.csv"}, cache_dir=config.cache_dir)    
    output0 = make_dataset(config, model.tokenizer, train_dataset['train'], True)
    eval_dataset = make_dataset(config, model.tokenizer, eval_dataset['dev'])
    #import pdb; pdb.set_trace()
    exmp_sim_score = None
    if config.use_cl_exmp:
        train_dataset, exmp_sim_score = output0
    else:
        train_dataset= output0


    test_dataset = load_dataset(path='csv', name=config.task_name, data_files={'test':config.data_dir+"/{}_test.csv".format(config.task_name)}, cache_dir=config.cache_dir, split="test[:]") #to be fix
    test_dataset = make_dataset(config, model.tokenizer, test_dataset)
    #test_dataset = output[0] if config.use_cl_exmp else output
    
    if config.option == 'train':
        train_process(config, train_dataset, eval_dataset, test_dataset, exmp_sim_score, model)
    elif config.option == "test":
        model.model.load_state_dict(os.path.join(config.output_dir, 'pytorch_model.bin'))

        test_dataset = load_dataset(path='csv', name=config.task_name, data_files={'test':config.data_dir+"/{}_test.csv".format(config.task_name)}, cache_dir=config.cache_dir)
        test_dataset = make_dataset(config, model.tokenizer, test_dataset['test'])
        #test_dataset = output[0] if config.use_cl_exmp else output
        test_process(config, test_dataset, model)
    

    #import pdb; pdb.set_trace()
