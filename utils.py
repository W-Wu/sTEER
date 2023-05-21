import numpy as np
import torch
import json
import logging
import random
from pyannote.core import Annotation, Segment

logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_json(json_file):
    f=open(json_file,'r')
    text=f.read()
    dic=json.loads(text)
    return dic

def dump_json(json_dict,json_file):
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

def Rttm2dic(rttm_file):
    dic={}
    f = open(rttm_file,'r')
    lines = f.readlines()
    
    for line in lines:
        if line.startswith('SPEAKER'):
            tmp = line.split()
            diag_id =tmp[1]
            if diag_id not in dic:
                dic[diag_id] = Annotation()
            start = float(tmp[3])
            end = start+float(tmp[4])
            spkr = tmp[-3]
            dic[diag_id][Segment(start,end),diag_id]=spkr

    return dic


class MetricStats_acc:
    def __init__(self):
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.correct = 0.0
        self.total = 0.0
        self.ids = []
        self.summary = {}


    def append(self, ids, predictions, targets):
        """Store a predictions and targets for later use
        """
        self.ids.extend(ids)
        idx = torch.argmax(predictions.detach(),dim=-1)
        cor = sum(idx == targets.detach()).item()
        N = len(targets)
        self.correct+=cor
        self.total+=N

    def summarize(self):
        """Compute statistics using a full set of results
        """
        scores = self.correct/self.total
        return scores
