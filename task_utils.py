import random
import numpy as np
from sklearn import metrics
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_task_info(task):
    pass

metrics_list = ['acc', 'f1']

#modify this dictionary for new task
task_dict = {
    'imdb':{
        'num_label':2,
        'label_ids':{'positive':0, 'negative':1},
        'max_len':512,
        'template':'the movie is <mask>, <sent>',
        'label_words':[['Ġfantastic', 'Ġwonderful'], ['Ġterrible', 'Ġbad']],
        'metrics':['acc']
    },

}

    
def set_config_for_task(config):
    #set labels, template
    #set metrics
    task = config.task_name.lower()
    task_list = ['imdb']
    assert task in task_list, "task '{}' not seen, use task in {} or modify code for new task".format(task, task_list)
    dict_task = task_dict[task]
    if config.metrics is None: config.metrics = dict_task['metrics']
    if config.num_label is None: config.num_label = dict_task['num_label']
    if config.max_len is None: config.max_len = dict_task['max_len']
    if config.template is None: config.template = dict_task['template']
    if config.label_words is None: 
        num_label_words = min(config.num_label_words, len(dict_task['label_words']))
        config.label_words = [words[:num_label_words] for words in dict_task['label_words']]
    if config.label_ids is None: config.label_ids = dict_task['label_ids']
    
def compute_metrics(preds, labels, metrics):
    result = None
    if metrics == 'acc':
        result = accuracy_score(labels, preds)
    elif metrics == 'f1':
        result = f1_score(labels, preds, average="macro")
    else:
        raise RuntimeError('no metric {}, must in this list {}'.format(metrics, metrics_list))
    return result
