import json

def read_json(json_file):
    f=open(json_file,'r')
    text=f.read()
    dic=json.loads(text)
    return dic

def dump_json(json_dict,json_file):
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

test_id=5
SAMPLERATE=16000
output_dir='../data/'
ses_wav_dic=read_json('../data/iemo_ses.json')
rttm_file = f'../exp-eval/save/sys_rttms/IEMO_test-{test_id}/oracle_cos_SC/sys_output.rttm'

f = open(rttm_file,'r')
lines = f.readlines()
dic = {}
for line in lines:
    if line.startswith('SPEAKER'):
        tmp = line.split()
        start = float(tmp[3])
        end = start+float(tmp[4])
        spkr = tmp[-3]
        diag_id = tmp[1][:-1]
        seg_id = f"{diag_id}-{round(start,4)}_{round(end,4)}"
        dic[seg_id] = {
            "wav": ses_wav_dic[diag_id]['fea_path'],
            "start": int(start*SAMPLERATE),
            "stop": int(end*SAMPLERATE),
            "spk_id": spkr,
        }    
dump_json(dic,output_dir+f'iemo-test-{test_id}-drz.json')
