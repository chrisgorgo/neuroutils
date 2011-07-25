'''
Created on 1 Apr 2011

@author: filo
'''
import nibabel.trackvis as tv
import nibabel as nb
import numpy as np
import cPickle

pkl_file = open('streamlines.pkl', 'rb')

streamlines = cPickle.load(pkl_file)
print type(streamlines)        
        
hdr = tv.empty_header()
nifti_filename = 'dtifit__FA.nii'
nii_hdr = nb.load(nifti_filename).get_header()
hdr['dim'] = np.array(nii_hdr.get_data_shape())
hdr['voxel_size'] = np.array(nii_hdr.get_zooms())
aff = np.eye(4)
aff[0:3,0:3] *= np.array(nii_hdr.get_zooms())
hdr['vox_to_ras'] = aff

print hdr['version']

for i in range(len(streamlines)):
    
    points_arr = streamlines[i][0]
    
    #invert y
    points_arr[:,1] = nii_hdr.get_data_shape()[1] - points_arr[:,1]
    
    #move to mm dim with 0,0,0 origin
    points_arr = points_arr*nii_hdr.get_zooms()
        
    streamlines[i] = (points_arr, None, None)
    
f='test.trk'    
fobj=open(f,'w')     
tv.write(fobj,streamlines,hdr)
fobj.close()