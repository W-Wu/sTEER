import pandas as pd
import jiwer
import numpy as np
import sys
sys.path.append('../')
from utils import *

ref_dic={}
iemocap_ref_file = '../data/iemo_trans_organized-noPunc_all.csv'
df = pd.read_csv(iemocap_ref_file, delimiter='\t', header=None, names=['utt_name','sentence'])

sentences = df.sentence.values.astype(str)
utt_name = df.utt_name.values.tolist()

for i in range(len(sentences)):
    diag_id = "_".join(utt_name[i].split('_')[:-1])
    if diag_id not in ref_dic:
        ref_dic[diag_id]={}
    spk_id = diag_id[:5]+'.'+utt_name[i].split('_')[-1][0]
    if spk_id not in ref_dic[diag_id]:
        ref_dic[diag_id][spk_id]=[]
    ref_dic[diag_id][spk_id].append(sentences[i])


transformation = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
]) 
print("test_id\tI\tD\tS\tH\tcpWER")
map_spk={}

test_id = 5
eval_folder='../exp'
asr_file = f'{eval_folder}/SR_test_outcome-{test_id}-drz.json'
asr_dic=read_json(asr_file)

spk_file =f"../data/iemo-test-{test_id}-drz.json"
spk_dic=read_json(spk_file)

concat_asr_dic={}
for utt_id, text in asr_dic.items():
    diag_id = utt_id.split('-')[0]
    if diag_id not in concat_asr_dic:
        concat_asr_dic[diag_id]={}
    spk_id = spk_dic[utt_id]['spk_id']
    if spk_id not in concat_asr_dic[diag_id]:
        concat_asr_dic[diag_id][spk_id]=[]
    concat_asr_dic[diag_id][spk_id].append(text)

WER_list_diag=[]
for diag_id in concat_asr_dic:
    for ref_spk in ref_dic[diag_id].keys():
        WER_list_tmp=[]
        for hyp_spk in concat_asr_dic[diag_id].keys():
            ref_list=" ".join(ref_dic[diag_id][ref_spk])
            asr_list=" ".join(concat_asr_dic[diag_id][hyp_spk])
            tmp_wer=jiwer.compute_measures(
            ref_list, 
            asr_list, 
            # truth_transform=transformation, # already processed
            hypothesis_transform=transformation
            )
            WER_list_tmp.append([ref_spk,hyp_spk,tmp_wer['insertions'],tmp_wer['deletions'],tmp_wer['substitutions'],tmp_wer['hits'],tmp_wer['wer']])
        WER_list_tmp.sort(key=lambda x: x[-1])
        WER_list_diag.append(WER_list_tmp[0])
        map_spk[WER_list_tmp[0][1]]=WER_list_tmp[0][0]
total_stats=np.array([x[2:-1] for x in WER_list_diag]).sum(axis=0)
# wer = float(S + D + I) / float(H + S + D)
I,D,S,H=total_stats
min_WER = float(S + D + I) / float(H + S + D)
    
print(f"{test_id}\t{I}\t{D}\t{S}\t{H}\t{min_WER}")

np.save(f'{eval_folder}/cp_map_spk.npy',map_spk)
