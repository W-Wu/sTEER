import os


iemo_root = "IEMOCAP/"
output_dir = '../data'

file_roots=[iemo_root+"Session"+str(i)+"/dialog/transcriptions" for i in range(1,6)]

file_paths=[]   

for file_dir in file_roots:
    for files in os.listdir(file_dir):  
        if os.path.splitext(files)[1] == '.txt':  
            file_paths.append(os.path.join(file_dir, files)) 

file_paths.sort()
c_raw_all={}
for i in range(0,len(file_paths)):
    file_path=file_paths[i]
    dddd=[]
    Flag=0
    (filepath,tempfilename) = os.path.split(file_path)
    (filename,extension) = os.path.splitext(tempfilename)
    f = open(file_path,'r')        
    line = f.readline()
    while line:
        if len(line.split(' ',2))>=3:
            uttname=line.split(' ')[0] 
            if 'Ses' in uttname and 'XX' not in uttname:
                line=line.split(' ',2)[2]   #remove utt name and time info, only keep sentence
                line=line.strip('\n')
                if uttname in c_raw_all:
                    c_raw_all[uttname]+=line
                else:
                    c_raw_all[uttname]=line
        line=f.readline()
assert len(c_raw_all) == 10039
with open(output_dir+'iemo_trans_raw_all.csv','w') as raw_all:
    for key in c_raw_all:
        raw_all.write(key+'\t'+c_raw_all[key]+'\n')
