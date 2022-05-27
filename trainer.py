from datetime import datetime
import logging
import os
from numpy import gradient
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from models import TransformerModelWrapper
from copy import deepcopy
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trainer')

def train_process(config, train_dataset:Dataset, eval_dataset:Dataset,  
                    test_dataset:Dataset,  exmp_sim_score:torch.Tensor, 
                    model:TransformerModelWrapper, ):

    if config.async_test:
        test_lock = threading.Lock()

    for stage in range(config.train_stage):
    # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.model.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.model.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]    
        embedding_parameters = [{'params': [p for p in model.model.model.get_input_embeddings().parameters()],
                                    'weight_decay': 0.0}]
        #未定义dp      
        learning_rate = config.learning_rate
        if stage == 1:
            handle = model.encoder.add_embed_hook(model.model.model)
            optimizer_grouped_parameters = embedding_parameters  
            learning_rate = config.embedding_learning_rate
            #exmp_sim_score = None #to be fix
        else:
            #if stage == 2: exmp_sim_score = None 

            # Training stage 0 / 2: optimize all model weights with different learning rates
            # This is used when training LM ONLY!
            #handle = model.encoder.add_reverse_hook((model.model.model))
            embedding_parameters = [{'params': [p for p in model.model.model.get_input_embeddings().parameters()],
                                        'weight_decay': 0.0}]
            optimizer_grouped_parameters[0] = {'params': [p for n, p in model.model.model.named_parameters()
                                                            if not any(nd in n for nd in no_decay + ['word_embeddings'])],
                                                'weight_decay': config.weight_decay}
            # Mask out gradients of tokens unrelated with prompt / label
            if config.fix_other_embeddings == True: #maybe a problem
                #handle = model.encoder.add_embed_hook(model.model.model)
                handle = model.encoder.add_reverse_hook((model.model.model))
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=config.adam_epsilon)
        
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size)
        t_total =  (len(train_dataloader) // config.grad_accu + bool(len(train_dataloader) % config.grad_accu)) * config.num_train_epochs
        num_warmup_steps = t_total * config.warmup_rate
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
        logger.info("scheduler params: warmup step: %d, total step: %d" % (num_warmup_steps, t_total))

        now = datetime.now()
        path_suffix = now.strftime('%m-%d_%H:%M:%S') + 'stage_%d' % stage    
        writer = SummaryWriter(log_dir=os.path.join(
            config.output_dir, "writer_logs", path_suffix))
        device = config.main_cuda if not config.no_cuda else 'cpu'
        model.to(device)
        cur_model = model
        if len(config.cuda_list) > 1: #Data Parallel
            cur_model.model = nn.DataParallel(cur_model.model, device_ids=config.cuda_list, output_device=config.main_cuda)
        global_step = 0 
        best_score = 0.0 
        train_iterator = trange(int(config.num_train_epochs), desc="Epoch")
        #import pdb; pdb.set_trace()
        for iter in train_iterator:
            epoch_step, train_loss = cur_model.train(iter, train_dataset, exmp_sim_score, optimizer, scheduler)
            global_step += epoch_step
            writer.add_scalar('train_loss', train_loss, global_step)
            if (iter+1) % config.num_logging_epochs == 0 or iter+1 == config.num_train_epochs:
                    logger.info("iteration %d, train loss: %.8f" % (iter, train_loss))
                
            if (iter+1) % config.num_eval_epochs == 0 or iter+1 == config.num_train_epochs:
                results = cur_model.eval(eval_dataset, config.metrics)
                logger.info("stage %d iteration %d, eval loss: %.4f, metrics: %s" % (stage, iter, results['eval_loss'], results['metrics']))
                metric_for_best_model = config.metrics[0]
                writer.add_scalar('eval_loss', results['eval_loss'], global_step)
                writer.add_scalars('eval_metrics', results['metrics'], global_step)

                if results['metrics'][metric_for_best_model] >= best_score:
                    best_score = results['metrics'][metric_for_best_model]
                    cur_model.save(config.output_dir+'/stage_{}'.format(stage), results, iter)

                if config.do_predict:

                    assert test_dataset is not None
                    if config.async_test:
                        test_lock.acquire()
                        Args = (config, test_dataset, cur_model, stage, iter, writer, global_step, test_lock)
                        test_thread = threading.Thread(target=Asynchronous_test_process, args=Args)
                        test_thread.start()
                    else:
                        results = cur_model.eval(test_dataset, config.metrics)
                        logger.info("stage %d iteration %d, test loss: %.4f, metrics: %s" % (stage, iter, results['eval_loss'], results['metrics']))
                        metric_for_best_model = config.metrics[0]
                        writer.add_scalars('test_metrics', results['metrics'], global_step)    
                        if config.save_result:
                            cur_model.save_result(config.output_dir, results, iter, True)             

        try:
            handle.remove()
        except Exception:
            pass
        assert test_dataset is not None
        if config.async_test:
            test_lock.acquire()
            Args = (config, test_dataset, cur_model, stage, iter, writer, global_step, test_lock)
            test_thread = threading.Thread(target=Asynchronous_test_process, args=Args)
            test_thread.start()
        else:
            results = cur_model.eval(test_dataset, config.metrics)
            logger.info("stage %d iteration %d, test loss: %.4f, metrics: %s" % (stage, iter, results['eval_loss'], results['metrics']))
            metric_for_best_model = config.metrics[0]
            writer.add_scalars('test_metrics', results['metrics'], global_step)    
            if config.save_result:
                cur_model.save_result(config.output_dir, results, iter, True)          
            
        cur_model.model = cur_model.model.module if hasattr(cur_model.model, 'module') else cur_model.model


        


def test_process(config, test_dataset:Dataset, model:TransformerModelWrapper):
    test_dataloader = DataLoader(test_dataset, batch_size=config.train_batch_size*2)
    best_score = 0.0 

    device = config.main_cuda if not config.no_cuda else 'cpu'
    model.to(device)
    #if len(config.cuda_list) > 1: #Data Parallel
    #    model = nn.DataParallel(model, device_ids=config.cuda_list)

    logger.info("Testing......")
    results = model.eval(test_dataset, config.metrics)
    logger.info("test over, test loss: %.4f, metrics: {}" % (results['eval_loss'], results['metrics']))
    if config.save_result:
        model.save_result(config.output_dir, results, 0, True)
        pass

def Asynchronous_test_process(config, test_dataset:Dataset, model:TransformerModelWrapper, stage:int, iter:int, writer:SummaryWriter, global_step:int, lock:threading.Lock):
    #assert config.main_cuda != config.async_cuda
    device = config.async_cuda if not config.no_cuda else 'cpu'
    async_model = deepcopy(model).to(device)
    async_model.model = async_model.model.module if hasattr(async_model.model, 'module') else async_model.model

    logger.info("Testing asynchronously......")
    results = async_model.eval(test_dataset, config.metrics, True)
    logger.info("asynchronous test over, stage %d iteration %d, test loss: %.4f, metrics: %s" % (stage, iter, results['eval_loss'], results['metrics']))
    writer.add_scalars('test_metrics', results['metrics'], global_step)
    if config.save_result:
        async_model.save_result(config.output_dir, results, iter, True)     

    del async_model
    lock.release()
    pass
