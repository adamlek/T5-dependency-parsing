from collections import defaultdict
from pickletools import float8
from random import shuffle
from typing import Counter
from IPython import embed
from data import *
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
from transformers.optimization import LambdaLR
import torch 
from curriculum import CurriculumDataloader, EvaluationDatasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda:3')
model_card = 'google/flan-t5-base'

def train(loader, full_competence, train_steps, run_name):
    writer = SummaryWriter(f'runs/{run_name}')
    
    tokenizer = T5Tokenizer.from_pretrained(model_card)
    model = T5ForConditionalGeneration.from_pretrained(model_card).to(device)
    
    scale, rel_step, warm_i, lr  = True, True, True, None
    # Boring
    opt = Adafactor(model.parameters(), 
                    scale_parameter=scale, 
                    relative_step=rel_step, 
                    warmup_init=warm_i, 
                    lr=lr)
    
    tokenizer.add_special_tokens({'additional_special_tokens':loader.deprels})
    model.resize_token_embeddings(len(tokenizer))
    
    model.train()
    for i in tqdm(range(train_steps)):
        batch = loader.get_batch()
        
        input = tokenizer.batch_encode_plus(batch.forms, 
                                            padding=True, 
                                            add_special_tokens=True,
                                            is_split_into_words=True, 
                                            return_tensors='pt')
        
        target = tokenizer.batch_encode_plus(batch.target, 
                                             padding=True, 
                                             add_special_tokens=True, 
                                             is_split_into_words=True, 
                                             return_tensors='pt')
        
        output = model(input_ids=input['input_ids'].to(device), 
                       attention_mask=input['attention_mask'].to(device), 
                       labels=target['input_ids'].to(device))
        
        output.loss.backward()
        opt.step()
        opt.zero_grad()
        
        # update competence
        if loader.scoring != 'random':
            if i+1 == full_competence:
                loader.reset_data()
            elif i+1 < full_competence:
                loader.update_competence(i, 'linear')
            else:
                pass
            
        writer.add_scalar('Competence', loader.competence, i)
        writer.add_scalar('Loss', output.loss.item(), i)
    
    model.save_pretrained(f'models/{run_name}')
    tokenizer.save_pretrained(f'models/tokenizer')
    print('Model and tokenizer saved')


def run_eval_model(loader, run_name):
    
    eval_file = open(f'eval_{run_name}.csv', '+w')
    
    # load models
    model = T5ForConditionalGeneration.from_pretrained(f'./models/{run_name}').to(device)
    tokenizer = T5Tokenizer.from_pretrained(f'./models/tokenizer')
    model.eval()
    
    with torch.no_grad():
        for i, e_batch in enumerate(loader.generate_eval_batches('dev')):
            print(i, end='\r')
            
            input = tokenizer.batch_encode_plus(e_batch.forms, 
                                                padding=True, 
                                                add_special_tokens=True,
                                                is_split_into_words=True, 
                                                return_tensors='pt').to(device)
            
            output = model.generate(input_ids=input['input_ids'], 
                                    attention_mask=input['attention_mask'],
                                    max_length=512,
                                    top_k=20, # 10?
                                    do_sample=True, 
                                    num_beams=2)
            
            for i, pred in enumerate(tokenizer.batch_decode(output)):
                pred = pred.split('</s>')[0].replace('<pad>', '')
                _gold = ' '.join(e_batch.target[i])
                eval_file.write('\t'.join([_gold, pred])+'\n')
    
    eval_file.close()
    return
                
def score_file(file_name):
    uas, las = [], []
    total, errs = 0, 0
    corr_len = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            gold, pred = line.rstrip().split('\t')
            
            # fix gold, for some reason it does wierd things
            gold = gold.lstrip().rstrip().replace(' :', ':').replace(' omp', 'omp').replace('; ', ';')
            pred = pred.lstrip().rstrip().replace(' :', ':').replace('; ', ';')
            
            corr_len.append(int(len(gold.split(' '))==len(pred.split(' '))))
            
            for w_gold, w_pred in zip(gold.split(' '), pred.split(' ')):
                try:
                    w_arc_gold, w_lbl_gold = w_gold.split(';')
                    w_arc_pred, w_lbl_pred = w_pred.split(';')
                    uas.append(int(w_arc_gold==w_arc_pred))
                    las.append(int(w_arc_gold==w_arc_pred and w_lbl_gold==w_lbl_pred))
                    total += 1
                except Exception as e:
                    errs += 1
    
    print('Correctly predicted length:', np.round(np.mean(corr_len)))
    print('Errors:', errs/total)
    print('UAS:', np.round(np.mean(uas), 3), 'LAS:', np.round(np.mean(las), 3))
    
def eval_all(train_steps, full_c):
    for sc in ['random', 'len', 'mhd', 'mdd']:
        print('Eval:: Scoring:', sc, 'Train steps:', train_steps, 'Full competence:', full_c)
        # stupid google
        run_name = f'{model_card}_full-competence={full_c},train={train_steps},scoring={sc}'.replace('google/','')
        
        loader = EvaluationDatasets('treebanks/', 
                                    'en_ewt-ud-dev.conllu', 
                                    'en_ewt-ud-test.conllu',
                                    batch_size=8)
        
        print('Evaluating:', run_name)
        run_eval_model(loader, run_name)
    
    # Evaluate all files
    eval_files(train_steps, full_c)

def train_model(sc, train_steps, full_c, run_name, batch_size=8):
    print('Train:: Scoring:', sc, 'Train steps:', train_steps, 'Full competence:', full_c)
    
    # create dataset loader
    loader = CurriculumDataloader('treebanks/', 
                                  'en_ewt-ud-train.conllu', 
                                  'en_ewt-ud-dev.conllu', 
                                  'en_ewt-ud-test.conllu',
                                  full_c, 
                                  batch_size=batch_size, 
                                  scoring=sc,
                                  prompt='Dependency parsing:')
    
    # run and save model
    train(loader, full_c, train_steps, run_name)       
    
def eval_files(train_steps, full_c):
    for sc in ['random', 'len', 'mhd', 'mdd']:
        run_name = f'{model_card}_full-competence={full_c},train={train_steps},scoring={sc}'.replace('google/', '')
        print(run_name)
        score_file(f'eval_{run_name}.csv')
    

if __name__ == '__main__':
    train_steps, full_c = 14000, 13000
    for sc in ['random', 'len', 'mhd', 'mdd']:
        run_name = f'{model_card}_full-competence={full_c},train={train_steps},scoring={sc}'.replace('google/','')
        train_model(sc, train_steps, full_c, run_name)
        
    eval_all(train_steps, full_c)
    

