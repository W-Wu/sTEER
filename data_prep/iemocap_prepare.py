"""
Data preparation.

*Author:
    Wen 2023
"""

import os
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files
import json
import numpy as np
import soundfile as sf
from pyannote.core import Annotation

from utils import *

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_iemocap(data_folder,
                    save_folder,
                    trans_file,
                    test_id,
                    max_subseg_dur,
                    overlap,
                    ):
    data_folder = data_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    text_dict = text_to_dict([trans_file])
    wav_lst =[]

    for ses_id in range(1,6):
        wav_lst += get_all_files(os.path.join(data_folder, f"Session{ses_id}","sentences/wav"), match_and=[".wav"])
    assert len(wav_lst) == 10039

    # diag level split -- ses blind
    traincv_list=[diag_id for diag_id in ses_wav_dic.keys() if int(diag_id[4])!=test_id]
    test_list=[diag_id for diag_id in ses_wav_dic.keys() if int(diag_id[4])==test_id]
    train_list=[x for i,x in enumerate(traincv_list) if i%5 !=0]
    cv_list=[x for i,x in enumerate(traincv_list) if i%5 ==0]
    assert len(test_list)+len(train_list)+len(cv_list)==151

    test_wav_lst = [x for x in wav_lst if "_".join(x.split('/')[-1].split('_')[:-1]) in test_list]
    train_wav_lst = [x for x in wav_lst if "_".join(x.split('/')[-1].split('_')[:-1]) in train_list]
    valid_wav_lst = [x for x in wav_lst if "_".join(x.split('/')[-1].split('_')[:-1]) in cv_list]
    assert len(test_wav_lst)+len(train_wav_lst)+len(valid_wav_lst) == 10039


    iemo_seg_dic_train={x.split('/')[-1].split('.')[0]:seg_dic[x.split('/')[-1].split('.')[0]] for x in train_wav_lst}
    iemo_seg_dic_valid={x.split('/')[-1].split('.')[0]:seg_dic[x.split('/')[-1].split('.')[0]] for x in valid_wav_lst}
    iemo_seg_dic_test={x.split('/')[-1].split('.')[0]:seg_dic[x.split('/')[-1].split('.')[0]] for x in test_wav_lst}
    assert len(iemo_seg_dic_train)+len(iemo_seg_dic_valid)+len(iemo_seg_dic_test) == 10039

    create_json(save_folder, test_wav_lst, text_dict,f"iemo-test-{test_id}-MTL")
    create_json(save_folder, train_wav_lst, text_dict,f"iemo-train-{test_id}-MTL")
    create_json(save_folder, valid_wav_lst, text_dict,f"iemo-valid-{test_id}-MTL")

    ref_rttm_dir=os.path.join(save_folder,'ref_rttms')
    if not os.path.exists(ref_rttm_dir):
            os.makedirs(ref_rttm_dir)

    meta_rttm_dir=os.path.join(save_folder,'metadata')
    if not os.path.exists(meta_rttm_dir):
            os.makedirs(meta_rttm_dir)

    # reference rttm for diariasation evaluation
    gen_ref_RTTM(test_list,f"{ref_rttm_dir}/fullref_iemo-test-{test_id}-VAD.rttm")

    # During testing, dialogue segmented into windows for VAD
    gen_meta(test_list,f"{meta_rttm_dir}/iemo-test-{test_id}-DiagAll_{max_subseg_dur}-{overlap}.json",max_subseg_dur,overlap)

        


def create_json(save_folder, wav_lst, text_dict, split):
    json_file = os.path.join(save_folder, split + ".json")
    if os.path.exists(json_file):
        print("Json file %s already exists, not recreating." % json_file)
        return
    
    json_dict = {}

    msg = "Creating %s..." % (json_file)
    print(msg)

    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:
        utt_id = wav_file.split("/")[-1].replace(".wav", "")
        ses_id = utt_id.split('_')[0]
        spk_id = ses_id[:-1]+'.'+utt_id.split('_')[-1][0]
        wrds = text_dict[utt_id]

        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        json_dict[utt_id] = {
            "wav": wav_file,
            "duration": signal.shape[0], 
            "spk_id": spk_id,
            "start": seg_dic[utt_id]["wav"]['start'],
            "stop": seg_dic[utt_id]["wav"]['stop'],
            "wrd": str(" ".join(wrds.split("_")))
        }

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    print(f"{json_file} successfully created!")


def text_to_dict(text_lst):
    text_dict = {}
    for file in text_lst:
        with open(file, "r") as f:
            for line in f:
                line_lst = line.strip().split("\t")
                text_dict[line_lst[0]] = "_".join(line_lst[1:])
    return text_dict


def get_RTTM_per_rec(segs, spkrs_list, rec_id):
    """Prepares rttm for each recording
    """
    rttm = []

    # Prepare header
    for spkr_id in spkrs_list:
        # e.g. SPKR-INFO ES2008c 0 <NA> <NA> <NA> unknown ES2008c.A_PM <NA> <NA>
        line = (
            "SPKR-INFO "
            + rec_id
            + " 0 <NA> <NA> <NA> unknown "
            + spkr_id
            + " <NA> <NA>"
        )
        rttm.append(line)

    # Append remaining lines
    for row in segs:
        # e.g. SPEAKER ES2008c 0 37.880 0.590 <NA> <NA> ES2008c.A_PM <NA> <NA>

        if float(row[1]) < float(row[0]):
            msg1 = (
                "Possibly Incorrect Annotation Found!! transcriber_start (%s) > transcriber_end (%s)"
                % (row[0], row[1])
            )
            msg2 = (
                "Excluding this incorrect row from the RTTM : %s, %s, %s, %s"
                % (
                    rec_id,
                    row[0],
                    str(round(float(row[1]) - float(row[0]), 4)),
                    str(row[2]),
                )
            )
            print(msg1)
            print(msg2)
            continue

        line = (
            "SPEAKER "
            + rec_id
            + " 0 "
            + str(round(float(row[0]), 4))
            + " "
            + str(round(float(row[1]) - float(row[0]), 4))
            + " <NA> <NA> "
            + str(row[2])
            + " <NA> <NA>"
        )
        rttm.append(line)

    return rttm


def gen_ref_RTTM(splits_list,rttm_file):
    ses_seg={}
    ses_spk={}
    if os.path.exists(rttm_file):
        # logger.info("Csv file %s already exists, not recreating." % csv_file)
        print("%s Already exists, not recreating." % rttm_file)
        return

    for diag_id in splits_list:
        ref_annotation = Annotation()
        ref_annotation=ref_annotation.from_json(ref_dic[diag_id])#.support()
        diag_id +='_'    #adding '_' to distinguisth Ses05M_script01_1b from Ses05M_script01_1
        for segment, track, label in ref_annotation.itertracks(yield_label=True):
            start,stop =segment
            spk_id=label
            if diag_id in ses_seg:
                ses_seg[diag_id]+=[[start,stop,spk_id]]
                
                ses_spk[diag_id].add(spk_id)
            else:
                ses_seg[diag_id]=[[start,stop,spk_id]]
                ses_spk[diag_id]=set([spk_id])
            assert len(ses_spk[diag_id])<=2, [diag_id,segment, label,ses_spk[diag_id],spk_id]

    RTTM = []
    for diag,segs in ses_seg.items():
        segs.sort(key=lambda x: float(x[0]))
        rttm_per_rec = get_RTTM_per_rec(segs, ses_spk[diag], diag)
        RTTM += rttm_per_rec

    # Write one RTTM as groundtruth. For example, "fullref_eval.rttm"
    with open(rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s\n" % item)
    print(f"{rttm_file} successfully created!")
    return rttm_file

def gen_RTTM(splits_dic,rttm_file):
    if os.path.exists(rttm_file):
        print("RTTM file %s already exists, not recreating." % rttm_file)
        return
    
    ses_seg={}
    ses_spk={}

    for utt_name,v in splits_dic.items():
        diag_id = "_".join(utt_name.split('_')[:-1])+'_'   # adding '_' to distinguisth Ses05M_script01_1b from Ses05M_script01_1
        ses_id = utt_name.split('_')[0]
        spk_id = ses_id[:-1]+'.'+utt_name.split('_')[-1][0]
        start = v["wav"]['start']/SAMPLERATE
        stop = v["wav"]['stop']/SAMPLERATE
        assert len(spk_id)>1, [diag_id,utt_name,utt_name,spk_id]
        if diag_id in ses_seg:
            ses_seg[diag_id]+=[[start,stop,spk_id]]
            ses_spk[diag_id].add(spk_id)
        else:
            ses_seg[diag_id]=[[start,stop,spk_id]]
            ses_spk[diag_id]=set([spk_id])
        assert len(ses_spk[diag_id])<=2, [diag_id,utt_name,utt_name,ses_spk[diag_id],spk_id]

    RTTM = []
    for diag,segs in ses_seg.items():
        segs.sort(key=lambda x: float(x[0]))
        rttm_per_rec = get_RTTM_per_rec(segs, ses_spk[diag], diag)
        RTTM += rttm_per_rec

    with open(rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s\n" % item)
    print(f"{rttm_file} successfully created!")


def gen_meta(splits_list,meta_path,max_subseg_dur,overlap):
    if os.path.exists(meta_path):
        print("Meta file %s already exists, not recreating." % meta_path)

    shift = max_subseg_dur - overlap
    # Create JSON from subsegments
    json_dict = {}
    for diag_id in splits_list:
        wav_file_path = ses_wav_dic[diag_id]["fea_path"]
        signal, fs = torchaudio.load(wav_file_path)
        signal = signal.squeeze(0)
        start = 0
        stop = len(signal)/SAMPLERATE

        seg_dur = stop-start

        if seg_dur > max_subseg_dur:
            num_subsegs = int(seg_dur / shift)
            # Now divide this segment (new_row) in smaller subsegments
            for i in range(num_subsegs):
                subseg_start = start + i * shift
                # make sure each segment is dur length to avoid short segments
                if subseg_start + max_subseg_dur - 0.01 > stop:
                    subseg_end = stop
                    subseg_start = stop - max_subseg_dur + 0.01
                else:
                    subseg_end = subseg_start + max_subseg_dur - 0.01

                json_dict[f"{diag_id}-{round(subseg_start,4)}_{round(subseg_end,4)}"] = {
                    "wav": {
                        "file": wav_file_path,
                        "start": int(subseg_start * SAMPLERATE),
                        "stop": int(subseg_end * SAMPLERATE),
                    },
                }
                # Break if exceeding the boundary
                if subseg_end >= stop:
                    break
        else:
            json_dict[f"{diag_id}-{round(start,4)}_{round(stop,4)}"] = {
                "wav": {
                    "file": wav_file_path,
                    "start": int(start * SAMPLERATE),
                    "stop": int(stop * SAMPLERATE),
                },
            }
        
    with open(meta_path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    msg = "%s JSON prepared" % (meta_path)
    print(msg)

def iemocap_segmentation(iemo_root,output_dir):
    json_file=output_dir+'iemocap_segmentation.json'
    if os.path.exists(json_file):
        print("%s already created, loading." % json_file)
        return read_json(json_file)
    file_roots=[iemo_root+"Session"+str(i)+"/dialog/transcriptions" for i in range(1,6)]
    file_paths=[]   
    for file_dir in file_roots:
        for files in os.listdir(file_dir):  
            if os.path.splitext(files)[1] == '.txt':  
                file_paths.append(os.path.join(file_dir, files)) 
    seg_dic={}
    for i in range(0,len(file_paths)):
            file_path=file_paths[i]
            f = open(file_path,'r')        
            line = f.readline()
            while line:
                if len(line.split(' ',2))>=3:
                    uttname=line.split(' ')[0] 
                    if 'Ses' in uttname and 'XX' not in uttname:
                        utt_dic = {}
                        _,time,text = line.split(' ',2)
                        text=text.strip('\n')
                        start,end=time[1:-2].split('-')
                        utt_dic['start']=float(start)
                        utt_dic['end']=float(end)
                        utt_dic['text']=text
                        if uttname in seg_dic:
                            raise RuntimeError("uttname already exists")
                        else:
                            seg_dic[uttname]=utt_dic

                line=f.readline()
    assert len(seg_dic) == 10039
    with open(json_file, mode="w") as json_f:
        json.dump(seg_dic, json_f, indent=2)
    return seg_dic

def iemo_ses(iemo_root,seg_ref_dic,output_dir):
    json_file=os.path.join(output_dir,'iemo_ses.json')
    if os.path.exists(json_file):
        print("%s already created, loading." % json_file)
        return read_json(json_file)
    ses_wav_dic={}
    for k,v in seg_ref_dic.items():
        diag_id = "_".join(k.split('_')[:-1])
        ses_id = k[4]
        if diag_id not in ses_wav_dic:
            wav_path = f'{iemo_root}/Session{ses_id}/dialog/wav-SingleChannel/{diag_id}.wav'
            sig, sr = sf.read(wav_path)
            assert sr==16000
            ses_wav_dic[diag_id]={'fea_path':wav_path,'duration':len(sig)}
    assert len(ses_wav_dic)==151
    
    with open(json_file, mode="w") as json_f:
        json.dump(ses_wav_dic, json_f, indent=2)
    return ses_wav_dic

def iemo_ses_segs(seg_ref_dic,ses_wav_dic,output_dir):
    json_file=os.path.join(output_dir,'iemo_ses_segs-all.json')
    if os.path.exists(json_file):
        print("%s already created, loading." % json_file)
        return read_json(json_file)
    ses_seg_dic={}
    for k,v in seg_ref_dic.items():
        diag_id = "_".join(k.split('_')[:-1])
        wav_path=ses_wav_dic[diag_id]['fea_path']
        start = seg_ref_dic[k]['start']
        end = seg_ref_dic[k]['end']
        ses_seg_dic[k]={"wav": {
                "file":wav_path,
                "duration":round(end-start, 3),
                "start":int(start*16000),
                "stop":int(end*16000),
        }
        }
    with open(json_file, mode="w") as json_f:
        json.dump(ses_seg_dic, json_f, indent=2)
    return ses_seg_dic


if __name__ == "__main__":
    
    iemo_root = "IEMOCAP/"
    output_dir = '../data/'
    seg_ref_dic=iemocap_segmentation(iemo_root,output_dir)
    ses_wav_dic=iemo_ses(iemo_root,seg_ref_dic,output_dir)
    seg_dic=iemo_ses_segs(seg_ref_dic,ses_wav_dic,output_dir)
    ref_dic=read_json('../data/iemo_VAD_lab.json')

    for test_id in range(1,6):
        if test_id !=5: continue
        prepare_iemocap(data_folder=iemo_root,
                trans_file='../data/iemo_trans_organized-noPunc_all.csv',
                save_folder=output_dir,
                test_id = test_id,
                max_subseg_dur=3,
                overlap=2
)
