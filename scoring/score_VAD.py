# Compute MSR and FAR for VAD
import numpy as np
from pyannote.core import Annotation, Segment
import json
from pyannote.metrics.detection import DetectionAccuracy

from utils import *

def get_diag_order(x):
  # Ses01F_impro02-0_2.99
  diag_dic={}
  for seg_id,v in x.items():
    diag_id = seg_id.split('-')[0]
    start, end = seg_id.split('-')[-1].split('_')
    if diag_id not in diag_dic:
      diag_dic[diag_id]=[]
    diag_dic[diag_id].append([seg_id,float(start),float(end)])
  for diag_id, v in diag_dic.items():
    diag_dic[diag_id].sort(key=lambda x: x[1])
  return diag_dic

def npy2Annote(x):
  y = [x[i]-x[i-1] for i in range(1,len(x))]
  start_ids = [i+1 for i,v in enumerate(y) if v == 1]
  end_ids = [i+1 for i ,v in enumerate(y) if v == -1]
  if x[0]==1:
    start_ids=[0]+start_ids
  if x[-1]==1:
    end_ids.append(len(x)-1)
  assert len(start_ids)==len(end_ids), (len(start_ids),len(end_ids))
  Annote = Annotation()
  for i in range(len(start_ids)):
    assert start_ids[i]<=end_ids[i],(start_ids[i],end_ids[i],x)
    start = start_ids[i] *0.02 # *320/16k
    end = end_ids[i]*0.02
    Annote[Segment(start,end)]='SPEECH'
  return Annote

def remove_overlap(npy_dic,order_dic):
  diag_dic={}
  for diag_id,order_list in order_dic.items():
    diag_vad=[]
    idx=0
    for i, v in enumerate(order_list):
      seg_id,start,end=v
      tmp = np.argmax(npy_dic[seg_id],axis=-1)
      assert len(tmp)==149, (i,seg_id,len(tmp)) # int(2.99*16000/320)
      if i == 0:
        diag_vad.extend(tmp[:100])
      elif i ==len(order_list)-1:
        cur_len = len(diag_vad)
        total_len = int(end*16000/320)
        seg_start = int(start*16000/320)
        assert seg_start<=cur_len
        diag_vad.extend(tmp[cur_len-seg_start-1:])
      else:
        diag_vad.extend(tmp[50:100])
    assert abs(total_len - len(diag_vad))<=1, (total_len , len(diag_vad))
    diag_dic[diag_id]=npy2Annote(diag_vad)
  return diag_dic


metric_DetectionAccuracy =  DetectionAccuracy(collar=0.25)


json_path = '../data/iemo_VAD_lab.json'
f=open(json_path,'r')
text=f.read()
ref_dic=json.loads(text)
assert len(ref_dic)==151


res_dic={'true negative':0,'true positive':0,'false positive':0,'false negative':0}
test_id = 5
npy_path = f'../exp/VAD_test_outcome-{test_id}-E-1.npy'
npy_dic = np.load(npy_path,allow_pickle=True).item()
order_dic = get_diag_order(npy_dic)
hpy_dic=remove_overlap(npy_dic,order_dic)


out_dic={}
for diag_id, v in hpy_dic.items():
  ref_annotation=Annotation()
  ref_annotation=ref_annotation.from_json(ref_dic[diag_id])
  ref_annotation=ref_annotation.support()
  hyp_annotation = hpy_dic[diag_id].support(collar=0.25)
  todel=[]
  for seg,track in hyp_annotation.itertracks():
    if seg.duration < 0.25:
      todel.append(seg)
  for seg in todel:
    del hyp_annotation[seg]

  out_dic[diag_id]=hyp_annotation.for_json()
  componet = metric_DetectionAccuracy.compute_components(ref_annotation, hyp_annotation)
  for k,v in componet.items():
    res_dic[k]+=v
    
dump_json(out_dic,npy_path.replace('.npy','-merge0.25.json'))


TN = res_dic['true negative']
TP = res_dic['true positive']
FP = res_dic['false positive']
FN = res_dic['false negative']
print("FalseAlarmRate:\t",FP/(TP+FN))
print("MissRate:\t",FN/(TP+FN))
