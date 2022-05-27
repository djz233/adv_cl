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
        'template': '<sent0> In summary, the film was <mask>.',
        #'template':'the movie is <mask>, <sent0>',
        #'template': '<sent0> All in all, it was <mask>.',
        #'label_words':[['Ġfantastic', 'Ġwonderful', 'Ġamazing', 'Ġexcellent'], ['Ġterrible', 'Ġbad', 'Ġawful', 'Ġhorrible']],
        #'label_words':[['Ġgreat','Ġexcellent',], ['Ġterrible', 'Ġhorrible',]],
        #'label_words': [['Ġgreat'], ['Ġterrible']]
        'label_words':[['Ġwonderful','Ġamazing',], ['Ġterrible', 'Ġbad',]],
        'dataset_column': ['sentiment', ['review']], #[label name is csv, [features in csv]]
        'metrics':['acc']
    },

    'sst2':{
        'num_label':2,
        #'label_ids':{'world':1, 'sport':2, 'business':3, 'sci/tech':4},
        'label_ids':{'0':0, '1':1},
        'max_len':110 ,
        #'template': '<sent> It was <mask>.', #to be fix
        'template': 'the movie is <mask>, <sent>',
        'label_words': [['irresistible'], ['pathetic']],
        'dataset_column': ['label', ['sentence']],
        'metrics':['acc']        
    },

    'ag_news':{
        'num_label':4,
        #'label_ids':{'world':1, 'sport':2, 'business':3, 'sci/tech':4},
        'label_ids':{1:0, 2:1, 3:2, 4:3},
        'max_len':128 ,
        'template': '<sent0> <sent1> This topic is about <mask>.', #to be fix
        'label_words': [['Ġpolitics'], ['Ġsports'],['Ġbusiness'], ['Ġtechnology']],
        'dataset_column': ['Class Index', ['Title', 'Description']],
        'metrics':['acc']
    },

    'dbpedia':{
        'num_label':14,
        #'label_ids':{'world':1, 'sport':2, 'business':3, 'sci/tech':4},
        'label_ids':{1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11, 13:12, 14:13},
        'max_len':128 ,
        'template': None, #to be fix
        'label_words': None,
        'dataset_column': ['label', ['title', 'content']],
        'metrics':['acc']        
    },

    'yahoo':{
        'num_label':10,
        #'label_ids':{'world':1, 'sport':2, 'business':3, 'sci/tech':4},
        'label_ids':{'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9,},
        'max_len':128 ,
        'template': None, #to be fix
        'label_words': None,
        'dataset_column': ['topic', ['question_title', 'question_content']],
        'metrics':['acc']        
    }
}

    
def set_config_for_task(config):
    #set labels, template
    #set metrics
    task = config.task_name.lower()
    task_list = ['imdb', 'sst2', 'dbpedia', 'ag_news', 'yahoo']
    assert task in task_list, "task '{}' not seen, use task in {} or modify code for new task".format(task, task_list)
    dict_task = task_dict[task]
    if hasattr(config, 'metrics'): config.metrics = dict_task['metrics']
    if hasattr(config, 'num_label'): config.num_label = dict_task['num_label']
    if hasattr(config, 'max_len'): config.max_len = dict_task['max_len']
    if hasattr(config, 'template') and config.template is None: config.template = dict_task['template']
    if hasattr(config, 'label_words') and config.label_words is None: 
        num_label_words = min(config.num_label_words, len(dict_task['label_words']))
        config.label_words = [words[:num_label_words] for words in dict_task['label_words']]
    if  hasattr(config, 'label_ids'): config.label_ids = dict_task['label_ids']
    if hasattr(config, 'dataset_column'): config.dataset_column = dict_task['dataset_column']
    
    
def compute_metrics(preds, labels, metrics):
    result = None
    if metrics == 'acc':
        result = accuracy_score(labels, preds)
    elif metrics == 'f1':
        result = f1_score(labels, preds, average="macro")
    else:
        raise RuntimeError('no metric {}, must in this list {}'.format(metrics, metrics_list))
    return result
