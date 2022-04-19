import imp
import logging
import os
from re import S
from turtle import forward
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, BertModel, BertPreTrainedModel,
                          RobertaForMaskedLM, RobertaModel, AutoModelForMaskedLM,
                          GPT2Tokenizer)
from typing import List, Dict
from task_utils import compute_metrics
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class ModelForPrompt(nn.Module):
    def __init__(self, config, model, ):
        super().__init__()
        self.config = config
        self.model = model
        #self.cl_temp = self.config.cl_temp
        self.sim = Similarity(temp=config.cl_temp)
        self.cl_pos_embedding = torch.rand((config.num_k*config.num_label, model.config.hidden_size), requires_grad=False) 
        self.cl_neg_embedding = torch.rand((config.num_k*config.num_label, model.config.hidden_size), requires_grad=False) 

    def cl_loss(self, mask_rep, cl_rep, labels):
        loss_fct = nn.CrossEntropyLoss()
        cos_sim = self.sim(mask_rep.unsqueeze(1).expand(
            -1, self.config.num_label, -1), cl_rep.unsqueeze(0))
        loss = loss_fct(cos_sim.squeeze(dim=0), labels)
        return loss


    def forward(self, input_ids, attention_mask, mlm_labels, labels, cl_examples=None, *args, **kwargs):
        B, L, hid_dim = mlm_labels.shape[0], mlm_labels.shape[1], self.model.config.hidden_size
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=mlm_labels, output_hidden_states=True)
        MLM_loss, pred_scores, hidden_states = outputs[0], outputs[1], outputs[2]
        total_loss = MLM_loss
        if cl_examples is not None:
            mask_pos = (mlm_labels != -100).unsqueeze(dim=-1).expand(B, L, hid_dim) #(bsz, L, dim)
            last_hidden_states = hidden_states[-1]
            mask_rep = last_hidden_states[mlm_labels != -100] #(bsz, dim)
            loss = self.cl_loss(mask_rep, cl_examples, labels)
            total_loss = self.config.alpha * loss + MLM_loss
        
        return total_loss, pred_scores, hidden_states
        
class LabelEncoder(object):
    def __init__(self, tokenizer, num_label:int, label_ids:dict):
        # Record prompt tokens
        pattern_token_set, pattern_token_indices = set(), []
        # RoBERTa tokenizer is initiated from GPT2Tokenizer,
        # and it tokenizes same words differently in different positions:
        # e.g.  'Hello world!' -> ['Hello', 'Ġworld', '!'];
        #       'Hello', 'world' -> ['Hello'], ['world']
        # So we need to add prefix space to simulate true situations
        kwargs = {'add_prefix_space': True} if isinstance(
            tokenizer, GPT2Tokenizer) else {}
        # Record label tokens
        label_token_ids = []
        '''
        for verbalizers, label_idx, in label_ids.items(): #["entailment", "not_entailment"]
            one_label_token_ids = []
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = tokenizer.convert_tokens_to_ids(verbalizer)
                assert verbalizer_id != tokenizer.unk_token_id, "verbalization was tokenized as <UNK>, verbalizer:{}".format(verbalizer)
                one_label_token_ids.append(verbalizer_id)
            label_token_ids.append(one_label_token_ids)
        '''

        assert len(pattern_token_set) < 50 and len(label_token_ids) < 49
        #print("prompt encoder")
        # Convert label to unused tokens
        # Note that `AlbertTokenizer` or `RobertaTokenizer` doesn't have a `vocab` attribute
        if hasattr(tokenizer, 'vocab') and '[unused0]' in tokenizer.vocab:
            # BERT
            self.label_convert = {token_id: tokenizer.vocab['[unused%s]' % idx]
                                    for idx, token_id in enumerate(range(num_label))}

        else:
            # ALBERT, RoBERTa
            start_idx = tokenizer.vocab_size - 100
            self.label_convert = {token_id: start_idx + idx
                                    for idx, token_id in enumerate(range(num_label))}

        # Convert mlm logits to cls logits
        self.vocab_size = tokenizer.vocab_size
        self.m2c_tensor = torch.tensor( 
            list(self.label_convert.values()), dtype=torch.long) #包含label unused token的tensor


    def init_embed(self, model, random_=False):
        w = model.get_input_embeddings().weight.data
        for origin_id, convert_id in self.label_convert.items():
            if random_:
                max_val = w[convert_id].abs().max()
                w[convert_id].uniform_(-max_val, max_val)
            else:
                w[convert_id] = w[origin_id] #to be fix


    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)
        #import pdb; pdb.set_trace()
        # Train certain tokens by multiply gradients with a mask
        #trainable_ids = list(self.pattern_convert.values()) + \
        trainable_ids =  list(self.label_convert.values())
        grad_mask = torch.zeros((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)
    
    def add_reverse_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        #trainable_ids = list(self.pattern_convert.values()) + \
        trainable_ids =  list(self.label_convert.values())
        grad_mask = torch.ones((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 0.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def get_replace_embeds(self, word_embeddings):
        return word_embeddings(self.lookup_tensor.to(word_embeddings.weight.device))

    def convert_mlm_logits_to_cls_logits(self, mlm_labels, logits):
        return torch.index_select(logits[mlm_labels != -100], -1, self.m2c_tensor.to(logits.device))


class TransformerModelWrapper(nn.Module):
    """A wrapper around a Transformer-based language model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hard_label = config.label_words

        # tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_fast=False)

        #self.pvp = PVPS[config.task_name](self, config.pattern_id)
        TransformersModel = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,            
            cache_dir=config.cache_dir if config.cache_dir else None,
            )
        
        self.model = ModelForPrompt(config, TransformersModel)
        '''
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i,
                          label in enumerate(self.config.label_list)}
        '''

        self.encoder = LabelEncoder(
            self.tokenizer, config.num_label,  config.label_words)#to be fix: too much param
        # Random init prompt tokens HERE!
        self.encoder.init_embed(self.model.model, random_=False)

        '''
        if config.device == 'cuda':
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
            # Use automatic mixed precision for faster training
            # self.scaler = GradScaler()
        '''
    
    def _generate_contrastive_embedding(self, inputs: torch.Tensor, epoch: int, dataset, is_posExam:bool=True) -> torch.Tensor:
        #generate all positive CL embedding for the epoch
        L = len(self.hard_label[0])
        hard_label = [label[epoch % L] for label in self.hard_label]
        mask_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pos_embedding = []
              
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids']
                label = batch['labels']
                #batch['output_hidden_states'] = True
                #idx = batch['idx']
                model = self.model.model.module if hasattr(
                    self.model, 'module') else self.model.model 
                mask_pos = (input_ids == mask_ids).long().argmax(dim=-1)
                if is_posExam == True: #generate positive examples, replace <mask> with hard lebel word
                    batch['input_ids'][0][mask_pos] = self.tokenizer.convert_tokens_to_ids(hard_label[label])  
                inputs = {
                    'input_ids': input_ids.to(model.device),
                    'attention_mask': batch['attention_mask'].to(model.device),
                    'output_hidden_states': True,
                }
                outputs = model(**inputs)
                pos_CLemb = outputs[-1][-1][0][mask_pos].squeeze(dim=0)
                pos_embedding.append(pos_CLemb)
        return torch.stack(pos_embedding, dim=0).detach()               
                

    def _generate_default_inputs(self, batch: Dict[str, torch.Tensor], pos_emb:torch.Tensor = None, neg_emb:torch.Tensor = None, sample_idx:torch.Tensor = None) -> Dict[str, torch.Tensor]:
        #add mlm_labels, cl_embedding(both positve and negative examples)
        inputs = batch
        input_ids = batch['input_ids'] #(bsz, max_len)?
        labels = batch['labels']
        converted_labels = torch.tensor(
            [self.encoder.label_convert[label.item()] for label in labels]
        )
        B = batch['input_ids'].shape[0]        
        model = self.model.module if hasattr(
            self.model, 'module') else self.model

        mask_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) 
        mask_pos = (input_ids == mask_ids).long()
        #import pdb; pdb.set_trace()
        mask_label = mask_pos * converted_labels.unsqueeze(-1).expand_as(mask_pos)
        mlm_labels = mask_label.masked_fill(mask_pos==0, -100).long()
        inputs['mlm_labels'] = mlm_labels
        
        
        #set cl_embedding
        if sample_idx is not None:
            dim_emb = pos_emb.shape[-1]
            cl_emb = torch.rand((B, self.config.num_label, dim_emb))
            for i, idx in enumerate(batch['idx']):
                i_th_sampled = sample_idx[idx]
                cl_emb[i] = torch.index_select(neg_emb.cpu(), dim=0, index=i_th_sampled) #neg emb to be fix, 第一次？
                label = batch['labels'][i]
                pos_idx = i_th_sampled[label]
                cl_emb[i][label] = pos_emb[pos_idx] #pos emb
            inputs['cl_examples'] = cl_emb
        
        if self.config.no_cuda == False:
            inputs = {k: t.cuda() if hasattr(t, 'cuda') else t
             for k, t in inputs.items()}

        return inputs

    #train for one epoch
    def train(self, epoch:int, dataset:Dataset, exmp_sim_score:torch.Tensor, optimizer: torch.optim.Optimizer, scheduler=None):
        #import pdb; pdb.set_trace()
        pos_emb, neg_emb = None, None
        cl_exmp_idx = None
        #choose cl examples index for generating cl example
        if self.config.use_cl_exmp == True:
            N = exmp_sim_score.shape[-1]
            perm = torch.randperm(N)
            cl_exmp_idx = exmp_sim_score[:,:, perm[0]]
            pos_emb = self._generate_contrastive_embedding(None, epoch, dataset)
            #For epoch 0, negative examples shoule be generated
            neg_emb = self.model.cl_neg_embedding if epoch else self._generate_contrastive_embedding(None, epoch, dataset, False)

        global_step = 0
        train_loss = 0.0 #sum format
        train_dataloader = DataLoader(dataset, self.config.train_batch_size, shuffle=True)
        total_batch = len(train_dataloader)
        self.model.train()
        for i, batch in enumerate(train_dataloader):
            output = self.mlm_train_step(batch, pos_emb=pos_emb, 
                                neg_emb=neg_emb, sample_idx=cl_exmp_idx)
            loss = output
            if self.config.grad_accu > 1:
                loss /= self.config.grad_accu
            try:
                loss.backward()
            except RuntimeError:
                pdb.set_trace()
            train_loss += loss.item() * len(batch['idx'])
            if (i+1) % self.config.grad_accu == 0 or i+1 == total_batch:
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                if scheduler is not None:   scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        return global_step, train_loss / total_batch, 
        #to be fix, add train_acc


    def eval(self, dataset:Dataset, metrics:List[str]=['acc']):
        eval_dataloader = DataLoader(dataset, self.config.train_batch_size * 2)
        total_batch = len(eval_dataloader)
        self.model.eval()
        preds = None
        all_idx, out_label_ids, question_ids = None, None, None
        all_masked_full_logits, all_masked_hidden_states = None, None
        eval_loss = 0.0     
        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            labels = batch['labels']
            indices = batch['idx']

            with torch.no_grad():
                loss, logits, masked_full_logits, masked_hidden_states = self.mlm_eval_step(
                    batch)
                if all_masked_hidden_states is None:
                    all_masked_full_logits = masked_full_logits.detach().cpu().numpy()
                    all_masked_hidden_states = masked_hidden_states.detach().cpu().numpy()
                else:
                    all_masked_full_logits = np.append(
                        all_masked_full_logits, masked_full_logits.detach().cpu().numpy(), axis=0)
                    all_masked_hidden_states = np.append(
                        all_masked_hidden_states, masked_hidden_states.detach().cpu().numpy(), axis=0)

                prediction_scores = logits.float()
                eval_loss += loss * len(indices)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(
                        question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        evaluate_results = {}
        for one_metrics in metrics:
            result = compute_metrics(preds.argmax(-1), out_label_ids, one_metrics) #to be fix
            evaluate_results[one_metrics] = result

        results = {
            "eval_loss": eval_loss / total_batch,
            'metrics': evaluate_results,
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            #'question_ids': question_ids,
            #'full_logits': all_masked_full_logits,
            #'masked_hidden_states': all_masked_hidden_states
        }

        return results

    def mlm_train_step(self, batch: Dict[str, torch.Tensor], pos_emb:torch.Tensor = None, neg_emb:torch.Tensor = None, sample_idx:torch.Tensor = None) -> torch.Tensor:
        inputs = self._generate_default_inputs(batch, pos_emb, neg_emb, sample_idx)   
        model = self.model.module if hasattr(
            self.model, 'module') else self.model
        outputs = model(**inputs)
        loss, pred_scores, hidden_states = outputs[0], outputs[1], outputs[2]
        if self.config.use_cl_exmp == True:
            last_layer_rep = hidden_states[-1]
            mask_reps = last_layer_rep[inputs['mlm_labels'] != -100]
            for idx, mask_rep in zip(batch['idx'], mask_reps):
                self.model.cl_neg_embedding[idx] = mask_rep.detach()
        return loss
        #to be fix, add train_acc

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]):
        inputs = self._generate_default_inputs(batch)
        model = self.model.module if hasattr(
            self.model, 'module') else self.model   
        outputs = model(**inputs)
        loss, pred_scores, hidden_states = outputs[0], outputs[1], outputs[2]

        ce_logits = self.encoder.convert_mlm_logits_to_cls_logits(inputs['mlm_labels'], pred_scores)
        masked_full_logits = pred_scores[inputs['mlm_labels'] >= 0]
        masked_hidden_states = hidden_states[-1][inputs['mlm_labels'] >= 0]

        return loss, ce_logits, masked_full_logits, masked_hidden_states 

    # 1. save eval/test results in text file; 
    # 2. for debugging or special experiments
    def _save_config_and_results(self, path, results):
        conf = json.dumps(self.config.__dict__, indent=3)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            f.write(conf)
        results_dict = results['metrics']
        resl = json.dumps(results_dict, indent=3)
        with open(os.path.join(path, 'results.json'), 'w') as f:
            f.write(resl)

    def save(self, path, results):
        logger.info("Saving trained model at %s..." % path)
        if self.config.debug:
            return        
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model     

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.config.save_result:
            self._save_config_and_results(path, results)    #to be fix

        embed_state = {
            "word_embeddings": model_to_save.model.get_input_embeddings().state_dict()            
        }
        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(embed_state, save_path_file)
