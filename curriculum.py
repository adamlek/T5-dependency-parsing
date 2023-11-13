from cmath import sqrt
from tkinter import Y
from IPython import embed
from typing import List
from random import shuffle, choices
import math
import numpy as np
from data import read_file
from collections import namedtuple
from toolz import take

class EvaluationDatasets():
    def __init__(self, base, dev, test, batch_size) -> None:
        super().__init__()
        self.data = {'dev': read_file(base+dev)[0],
                     'test': read_file(base+test)[0]}
        self.batch_size=batch_size
        
    def generate_eval_batches(self, eval_set):
        l = len(self.data[eval_set])
        eval_d = iter(self.data[eval_set])
        for _ in range(int(l/self.batch_size)+1):
            batch = list(take(self.batch_size, eval_d))
            if batch:
                yield self.construct_input(batch)
                
    def construct_input(self, batch):
        Batch = namedtuple('Batch', ['forms', 'target', 'mean_score', 'langs'])
        input_data = []
        target_data = []
        scoring = [0]
        langs = []
        
        for b in batch:
            input_data.append(['Parse the sentence:']+b['forms'])
            # [head_0;deprel_0, ..., head_n;deprel_n] 
            target_data.append(list(map(lambda x: x[0] + ';' + x[1], 
                                        list(zip(list(map(
                                            lambda x: str(x), b['heads'])), b['deprels'])))))
            langs.append(b['language'])
        return Batch(input_data, target_data, np.mean(scoring), langs)
        

class CurriculumDataloader():
    def __init__(self, base, data, dev, test, full_competence, batch_size=8, scoring='mhd', prompt='Parse the sentence:'):
        self.scoring = scoring
        self.prompt = prompt
        
        # For multi-lingual parsing:
        # individual dataset for each lang, then merge them in the sampling dataset after selection? 
        # theres an uneven distribution of samples for different competences given the language 
        # difference in mhd/mdd 
        self.data, train_labels = read_file(base+data)
        self.dev, dev_labels = read_file(base+dev)
        self.test, test_labels = read_file(base+test)
        self.deprels = list(train_labels.union(dev_labels).union(test_labels))
        self.c_update = {'sqrt':self.sqrt_update,
                         'linear':self.linear_update}

        self.initialize_data()
        self.initial_competence = 0.05
        self.sampling_dataset = self.data[:int(len(self.data)*self.initial_competence)]
        
        self.competence = 0.05
        self.steps_until_fully_competent = full_competence
        self.batch_size = batch_size
    
    def linear_update(self, step):
        c02 = self.initial_competence**2
        nc = (1-c02)/self.steps_until_fully_competent
        return min(1, step * nc + c02)
    
    def sqrt_update(self, step):
        c02 = self.initial_competence**2
        nc = (1-c02)/self.steps_until_fully_competent
        return min(1, math.sqrt(step * nc + c02))
    
    def update_competence(self, step, c_update='sqrt'):
        competence = self.c_update[c_update](step) # self.sqrt_update(step)
        self.update_sampling_dataset(self.competence, competence)
        
        self.competence = np.clip(competence, 0.0, 1.0)

    def update_sampling_dataset(self, prev, new):
        curr_len = int(len(self.data)*prev)
        new_len = int(len(self.data)*new)
        
        remove_add = new_len - curr_len
        self.sampling_dataset = self.sampling_dataset[remove_add:]
        self.sampling_dataset += self.data[curr_len:curr_len+remove_add]
        
    def reset_data(self):
        self.sampling_dataset = self.data
        
    def initialize_data(self):
        if self.scoring == 'random':
            shuffle(self.data)
        else:
            self.data = sorted(self.data, key=lambda x: x[self.scoring], reverse=False)
        
    def get_batch(self):
        if self.scoring == 'random':
            return self.construct_input(choices(self.data, k=self.batch_size))
        else:
            return self.construct_input(choices(self.sampling_dataset, k=self.batch_size))
    
    def construct_input(self, batch):
        Batch = namedtuple('Batch', ['forms', 'target', 'mean_score', 'langs'])
        input_data = []
        target_data = []
        scoring = []
        langs = []
        
        for b in batch:
            input_data.append([self.prompt]+b['forms'])
            # [head_0;deprel_0, ..., head_n;deprel_n] 
            target_data.append(list(map(lambda x: x[0] + '; ' + x[1], 
                                        list(zip(list(map(lambda x: str(x), b['heads'])), b['deprels'])))))
            scoring.append(b[self.scoring])
            langs.append(b['language'])
        return Batch(input_data, target_data, np.mean(scoring), langs)
    
if __name__ == '__main__':
    cd = CurriculumDataloader('treebanks/', 
                              'en_ewt-ud-train.conllu', 
                              'en_ewt-ud-dev.conllu',
                              'en_ewt-ud-dev.conllu',
                              full_competence=100)
    
    raise embed()