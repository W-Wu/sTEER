# VAD label --  speech segments according to word-level alignment
import os
import json
import re
from pyannote.core import Annotation, Segment

SIG_FRM=0.01
SR=16000
ENC_FRM=0.02

iemo_root = "IEMOCAP/"
output_dir = '../data/'

file_roots=[iemo_root+"Session"+str(i)+"/dialog/transcriptions" for i in range(1,6)]

file_paths=[]   

for file_dir in file_roots:
    for files in os.listdir(file_dir):  
        if os.path.splitext(files)[1] == '.txt':  
            file_paths.append(os.path.join(file_dir, files)) 


ses_all_seg={}  # break intermediate sil
cnt=0
for file_path in file_paths:
    ses_id = file_path.split('/')[-1].split('.')[0]
    ses_all_seg[ses_id]={}
    f = open(file_path,'r')        
    lines = f.readlines()
    for line in lines:
        if len(line.split(' ',2))>=3:
            uttname=line.split(' ')[0] 
            if 'Ses' in uttname and 'XX' not in uttname:
                _,time,text = line.split(' ',2)
                text=text.strip('\n')
                start,end=time[1:-2].split('-')
                cnt +=1
                wdseg_file = f'{iemo_root}/Session{ses_id[4]}/sentences/ForcedAlignment/{ses_id}/{uttname}.wdseg'
                try:
                    wdseg_f = open(wdseg_file,'r')   
                    lines = f.readlines()
                    #SFrm  EFrm    SegAScr Word
                    wdsegs = [x.split() for x in wdseg_f.readlines()[1:-1]]
                    # remove starting and ending silence
                    for i in range(len(wdsegs)):
                        if not wdsegs[i][-1].startswith('<'):
                            s = i
                            break
                    for i in range(len(wdsegs)-1,-1,-1):
                        if not wdsegs[i][-1].startswith('<'):
                            e = i
                            break
                    assert s<=e,(s,e,wdsegs)
                    tmp = wdsegs[s:e+1]
                    new_start = float(start) + float(tmp[0][0])*SIG_FRM
                    new_end = float(start) + float(tmp[-1][1])*SIG_FRM
                    new_wrd = " ".join([re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", x[-1]) for x in tmp])
                    new_wrd = new_wrd.replace("<sil>","")
                    
                    # if encounter intermediate silence, split to two segments
                    subseg_idx = 0
                    subseg_list=[]
                    for i in range(len(wdsegs)):
                        # SFrm,EFrm,SegAScr,Word=wdsegs[i]
                        if wdsegs[i][-1].startswith('<'):
                            if len(subseg_list)>0:
                                seg_name=uttname+'-'+str(subseg_idx)
                                seg_start = float(start) + float(subseg_list[0][0])*SIG_FRM
                                seg_end = float(start) + float(subseg_list[-1][1])*SIG_FRM
                                seg_wrd = " ".join([re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", x[-1]) for x in subseg_list])
                                subseg_list=[]
                                subseg_idx +=1 

                                ses_all_seg[ses_id][seg_name]={
                                            'start':round(float(seg_start),4),
                                            'end':round(float(seg_end),4),
                                            'wrd':seg_wrd,
                                            }
                        else:
                            # Ses03F_impro07_M030, Ses03M_impro03_M001
                            if len(subseg_list)>0:
                                assert int(wdsegs[i][0])-int(subseg_list[-1][1])==1, (subseg_list[-1],wdsegs[i])
                            subseg_list.append(wdsegs[i])
                except FileNotFoundError:
                    print(uttname)
                    ses_all_seg[ses_id][uttname]={
                                            'start':round(float(start),4),
                                            'end':round(float(end),4),
                                            'wrd':text,
                                            }
                    
                
assert len(ses_all_seg) ==151
assert cnt == 10039
json_file=output_dir+f'iemocap_segmentation_VAD.json'
with open(json_file, mode="w") as json_f:
    json.dump(ses_all_seg, json_f, indent=2)
