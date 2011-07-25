'''
Created on 1 Apr 2011

@author: filo
'''
import nibabel.trackvis as tv
import nibabel as nb
import numpy as np
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec,\
    InputMultiPath
from nipype.interfaces.traits import File
import os
import cPickle

filename = '/media/data/2010reliability/workdir/pipeline/seed/_subject_id_2402571160_20101210/_task_name_finger_foot_lips/probtractx/particle0'

f = open(filename, 'r')

tracts = []
cur_tract = []
cur_seed = (0,0,0)
second_half = False
seed = False
for line in f:
    if '.' in line:
        seed = False
    else:
        seed = True
    cur_point = [float(i) for i in line.split(" ")]
    #cur_point = tuple(np.dot(np.array(cur_point + (1,)),affine)[0:3])
    
    if seed and len(cur_tract) != 0 and second_half:
        #print "%s, %d\n"%(str(cur_seed), len(cur_tract))
        tracts.append((np.array(cur_tract), None, None))
        cur_tract = []
        second_half = False
    
    if len(cur_tract) == 0:
        cur_seed = cur_point
    elif cur_seed == cur_point:
        second_half = True
        continue
    
    if second_half:
        cur_tract.insert(0, cur_point)
    else:
        cur_tract.append(cur_point)
        
tracts.append((np.array(cur_tract), None, None))

output = open('streamlines.pkl', 'wb')

cPickle.dump(tracts, output)
output.close()
