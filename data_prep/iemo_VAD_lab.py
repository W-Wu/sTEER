import json
import numpy as np
from pyannote.core import Annotation, Segment

json_path='../data/iemocap_segmentation_VAD.json'
f=open(json_path,'r')
text=f.read()
seg_dic=json.loads(text)
assert len(seg_dic)==151



VAD_dic={}
for diag_id, utts in seg_dic.items():
    annotation = Annotation()
    for utt_name,v in utts.items():
        spk_id = utt_name[:5]+'.'+utt_name.split('_')[-1][0]
        annotation[Segment(float(v['start']), float(v['end'])),'_'] = spk_id
    VAD_dic[diag_id]=annotation.for_json()

assert len(VAD_dic)==151
output_dir='../data/'
with open(output_dir+'iemo_VAD_lab.json', mode="w") as json_f:
    json.dump(VAD_dic, json_f, indent=2)
