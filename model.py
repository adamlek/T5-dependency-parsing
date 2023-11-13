from transformers import T5EncoderModel
import torch
import torch.nn as nn


class T5Deps():
    def __init__(self, model_card) -> None:
        self.model = T5EncoderModel.from_pretrained(model_card)
        self.predict_head = nn.Linear(n, h)
        self.predict_labl = nn.Linear(n, l)
        
    def forward(self, input_ids, attention_mask):
        pass
    
    def predict(self, hidden_states):
        pass
    
    def loss(self, predictions, gold):
        pass