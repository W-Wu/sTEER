import os
import numpy as np

iemo_root='IEMOCAP/'
file_roots=[iemo_root+"Session"+str(i)+"/dialog/EmoEvaluation" for i in range(1,6)]
file_paths=[]   
for file_dir in file_roots:
    for files in os.listdir(file_dir):  
        if os.path.splitext(files)[1] == '.txt':  
            file_paths.append(os.path.join(file_dir, files))  
assert len(file_paths) == 151

output_dir='../data/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

lab_dic={}

for label_path in file_paths:
    Flag=0
    f = open(label_path,'r')
    line = f.readline()
    while line:
        if line.startswith('['):
            tmp = line.split()
            name=tmp[3]
            emo=tmp[4]
            lab_dic[name]=emo
        line=f.readline()
assert len(lab_dic)==10039
# print(set(lab_dic.values()))    
#{'neu', 'dis', 'hap', 'fea', 'fru', 'oth', 'xxx', 'ang', 'sur', 'sad', 'exc'}

from collections import Counter
print(Counter(lab_dic.values()))
# Counter({'xxx': 2507, 'fru': 1849, 'neu': 1708, 'ang': 1103, 'sad': 1084, 'exc': 1041, 'hap': 595, 'sur': 107, 'fea': 40, 'oth': 3, 'dis': 2})

for name,emo in lab_dic.items():
    if emo == 'exc':
        lab_dic[name]='hap'
    if lab_dic[name] not in ['xxx','neu','ang','sad','hap']:
        lab_dic[name]='oth'

assert len(lab_dic)==10039
print(Counter(lab_dic.values()))    
#Counter({'xxx': 2507, 'oth': 2001, 'neu': 1708, 'hap': 1636, 'ang': 1103, 'sad': 1084})



np.save(os.path.join(output_dir,'iemo-lab-MajCat6.npy'),lab_dic)
