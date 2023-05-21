# VAD lab: reference segmentation (having silence at the beginning/end/middle of the sentence) with intra-utterance frame-level speech/non-speech label
import os
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor,WavLMModel
import torch
import soundfile as sf


def frame_len(iemo_root):
    wav_folders=[f'{iemo_root}/Session{i}/sentences/wav' for i in range(1,6)]
    wav_paths = []
    for wav_folder in wav_folders:
        for diag_folder in os.listdir(wav_folder):
            wav_paths.extend([os.path.join(wav_folder,diag_folder,x) for x in os.listdir(os.path.join(wav_folder,diag_folder)) if x.endswith('.wav')] )

    assert len(wav_paths)==10039,len(wav_paths)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')
    model = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
    model.to(device)
    model.eval()

    len_dic={}
    with torch.no_grad():
        for wav_file in wav_paths:
            utt_name=wav_file.split('/')[-1].split('.')[0]
            audio_input,sample_rate = sf.read(wav_file)
            sig_len = len(audio_input)
            assert sample_rate==16000
            inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to(device)
            outputs = model(inputs['input_values'], output_hidden_states=True)
            hidden_states  = outputs.hidden_states
            len_dic[utt_name]=[sig_len,hidden_states[0].shape[1]]
    return len_dic


SR=16000
iemo_root = "IEMOCAP/"
output_dir = '../data/'

file_roots=[iemo_root+"Session"+str(i)+"/dialog/transcriptions" for i in range(1,6)]
file_paths=[]   
for file_dir in file_roots:
    for files in os.listdir(file_dir):  
        if os.path.splitext(files)[1] == '.txt':  
            file_paths.append(os.path.join(file_dir, files)) 
len_dic = frame_len(iemo_root)

vad_dic={}
for file_path in file_paths:
    ses_id = file_path.split('/')[-1].split('.')[0]
    f = open(file_path,'r')        
    lines = f.readlines()
    for line in lines:
        if len(line.split(' ',2))>=3:
            uttname=line.split(' ')[0] 
            if 'Ses' in uttname and 'XX' not in uttname:
                _,time,text = line.split(' ',2)
                text=text.strip('\n')
                start,end=time[1:-2].split('-')
                wav_file = f'{iemo_root}/Session{ses_id[4]}/sentences/wav/{ses_id}/{uttname}.wav'
                wdseg_file = f'{iemo_root}/Session{ses_id[4]}/sentences/ForcedAlignment/{ses_id}/{uttname}.wdseg'
                audio_input,sample_rate = sf.read(wav_file)
                sig_len = len_dic[uttname][-1]
                vad_lab=np.ones((sig_len))
                try:
                    wdseg_f = open(wdseg_file,'r')   
                    lines = f.readlines()
                    #SFrm  EFrm    SegAScr Word
                    wdsegs = [x.split() for x in wdseg_f.readlines()[1:-1]]
                    # remove starting and ending silence
                    for i in range(len(wdsegs)):
                        if wdsegs[i][-1].startswith('<'):
                            SFrm=int(wdsegs[i][0])//2 # *10ms * 16kHz /320
                            EFrm=int(wdsegs[i][1])//2

                            vad_lab[SFrm:EFrm]=0
                except FileNotFoundError:
                    print(uttname)
                    # wdseg file missing for Ses03F_impro07_M030, Ses03M_impro03_M001. Assuming all speech
                vad_dic[uttname]=vad_lab
                    
assert len(vad_dic) ==10039
np.save(output_dir+'iemo_VAD_lab-utt.npy',vad_dic)
