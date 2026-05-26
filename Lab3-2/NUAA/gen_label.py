import os
import random

PROJECT_DIR = "E:/ml/fas/"
NUAA_DATA_DIR = PROJECT_DIR + 'data/NUAA/'

real_face_train_file = NUAA_DATA_DIR + 'client_train_face.txt'
real_face_test_file = NUAA_DATA_DIR + 'client_test_face.txt'

spoof_face_train_file = NUAA_DATA_DIR + 'imposter_train_face.txt'
spoof_face_test_file = NUAA_DATA_DIR + 'imposter_test_face.txt'

train_file = NUAA_DATA_DIR + 'train.txt'
val_test_file = NUAA_DATA_DIR + 'val_test.txt'
val_file = NUAA_DATA_DIR + 'val.txt'
test_file = NUAA_DATA_DIR + 'test.txt'

if not os.path.exists(train_file):
    with open(train_file, 'w') as fw:
        content = []
        with open(real_face_train_file, 'r') as fr:
            real_lines = fr.readlines()
           
            for lines in real_lines:
                line = lines.split(' ')[0]
                line = line.replace('\\', '/')
                content.append(os.path.join(NUAA_DATA_DIR, 'ClientFace/' + line) + ',1')
        
        with open(spoof_face_train_file, 'r') as fr:
            real_lines = fr.readlines()
        
            for lines in real_lines:
                line = lines.split(' ')[0]
                line = line.replace('\\', '/')
                content.append(os.path.join(NUAA_DATA_DIR, 'ImposterFace/' + line) + ',0')
    
        random.shuffle(content)
        for line in content:
            fw.write(line + '\n')

if not os.path.exists(val_test_file):
    with open(val_test_file, 'w') as fw:
        content = []
        with open(real_face_test_file, 'r') as fr:
            real_lines = fr.readlines()
            
            for lines in real_lines:
                line = lines.split(' ')[0]
                line = line.replace('\\', '/')
                content.append(os.path.join(NUAA_DATA_DIR, 'ClientFace/' + line) + ',1')
        
        with open(spoof_face_test_file, 'r') as fr:
            real_lines = fr.readlines()
            
            for lines in real_lines:
                line = lines.split(' ')[0]
                line = line.replace('\\', '/')
                content.append(os.path.join(NUAA_DATA_DIR, 'ImposterFace/' + line) + ',0')
        
        random.shuffle(content)
        for line in content:
            fw.write(line + '\n')

if not os.path.exists(val_file):
    with open(val_test_file, 'r') as fr:
        lines = fr.readlines()
        line_num = len(lines)
        with open(val_file, 'w') as fw_val:
            with open(test_file, 'w') as fw_test:
                for idx, line in enumerate(lines):
                    if idx < 0.3 * line_num:
                        fw_val.write(line)
                    else:
                        fw_test.write(line)
                        
                        