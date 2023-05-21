import numpy as np
import json
import os

def read_json(json_file):
    f=open(json_file,'r')
    text=f.read()
    dic=json.loads(text)
    return dic

emo_lab = np.load('data/iemo-lab-MajCat6.npy',allow_pickle=True).item()
seg_dic =read_json('data/iemocap_segmentation_VAD.json')
map_spk = np.load('exp/cp_map_spk.npy',allow_pickle=True).item()

seg_emo_dic={}
for diag,utts in seg_dic.items():
    for seg_id, v in utts.items():
        utt_id = seg_id.split('-')[0]
        emo = emo_lab[utt_id]
        seg_name = f"{diag}-{v['start']}_{v['end']}"
        seg_emo_dic[seg_name]=emo

output_dir_ref = 'data/ref_emo_rttms/'
if not os.path.exists(output_dir_ref):
    os.makedirs(output_dir_ref)
output_dir_sys = 'exp-eval/'


test_id = 5
fullref_rttm = f'data/ref_rttms/fullref_iemo-test-{test_id}-VAD.rttm'
sysrttm_path = f'{output_dir_sys}/save/sys_rttms/IEMO_test-{test_id}/oracle_cos_SC/sys_output.rttm'

pred_emo_dic = np.load(f'exp/ER_test_outcome-CAT-{test_id}-drz.npy',allow_pickle=True).item()

# prepare full_ref
diags=[]
op_emo=[]
op_emo_spk=[]
f = open(fullref_rttm,'r')
lines = f.readlines()
for line in lines:
    if line.startswith('SPEAKER'):
        # SPEAKER Ses01F_impro02_ 0 8.4488 4.66 <NA> <NA> Ses01.F <NA> <NA>
        tmp = line.split()
        # print(tmp)
        diag_id = tmp[1][:-1]
        diags.append(tmp[1])
        start = float(tmp[3])
        end = start+float(tmp[4])
        spkr = tmp[-3]
        seg_name = f"{diag_id}-{round(start,4)}_{round(end,4)}"
        ref_emo = seg_emo_dic[seg_name]
        tmp[-3] = ref_emo
        op_emo.append(" ".join(tmp)+'\n')
        tmp[-3] = spkr+'+'+ref_emo
        op_emo_spk.append(" ".join(tmp)+'\n')
f.close()
# print(op_emo)
with open(output_dir_ref+f'fullref_iemo-test-{test_id}-TEER.rttm', 'w') as op:
    op.writelines(op_emo)
op.close()
with open(output_dir_ref+f'fullref_iemo-test-{test_id}-sTEER.rttm', 'w') as op:
    op.writelines(op_emo_spk)
op.close()

# prepare sys_output
op_emo=[]
op_emo_spk=[]
f = open(sysrttm_path,'r')
lines = f.readlines()
for line in lines:
    if line.startswith('SPEAKER'):
        # SPEAKER Ses01F_impro02_ 0 8.4488 4.66 <NA> <NA> Ses01.F <NA> <NA>
        tmp = line.split()
        diag_id = tmp[1][:-1]
        start = float(tmp[3])
        end = start+float(tmp[4])
        spkr = tmp[-3]
        seg_name = f"{diag_id}-{round(start,4)}_{round(end,4)}"
        ref_emo = pred_emo_dic[seg_name][0]
        tmp[-3] = ref_emo
        op_emo.append(" ".join(tmp)+'\n')
        tmp[-3] = map_spk[spkr]+'+'+ref_emo
        op_emo_spk.append(" ".join(tmp)+'\n')
f.close()

with open(output_dir_sys+f'ER_test_outcome-CAT-{test_id}-TEER.rttm', 'w') as op:
    op.writelines(op_emo)
op.close()
with open(output_dir_sys+f'ER_test_outcome-CAT-{test_id}-sTEER.rttm', 'w') as op:
    op.writelines(op_emo_spk)
op.close()
