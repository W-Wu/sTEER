import pandas as pd
import jiwer
import json

transformation = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
]) 

ref_dic={}
iemocap_ref_file = '../data/iemo_trans_organized-noPunc_all.csv'
df = pd.read_csv(iemocap_ref_file, delimiter='\t', header=None, names=['utt_name','sentence'])
sentences = df.sentence.values.astype(str)
utt_name = df.utt_name.values.tolist()

for i in range(len(sentences)):
    ref_dic[utt_name[i]]=sentences[i]

asr_file = f"../exp/SR_test_outcome-E-1.json"
f=open(asr_file,'r')
text=f.read()
asr_dic=json.loads(text)
ref_list=[]
asr_list=[]

for k in asr_dic.keys():
    ref_list.append(ref_dic[k])
    asr_list.append(asr_dic[k])
iemo_wer=jiwer.compute_measures(
    ref_list, 
    asr_list, 
    hypothesis_transform=transformation
)
print(iemo_wer['wer'])
