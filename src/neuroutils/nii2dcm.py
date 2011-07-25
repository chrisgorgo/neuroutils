'''
Created on 18 Jun 2011

@author: filo
'''
import dicom
#dicom.debug()
import os
import pylab
import nibabel
import numpy

#nifti_file = "/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_test_subject/_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/0rfinger_t_map_thr.img"
#original_dicom_dir = "/home/filo/tmp/13"
#dicom_outdir = "/tmp/dicom_out"

#original_dicom_dir = dicom_outdir


def nii2dcm(nifti_file, template_dicom_dir, output_dir, series_info_source_dicom, suffix, overlay=False, debug=False, description=None):
    nii_data = nibabel.load(nifti_file).get_data()
    print nii_data.shape
    n_slices = len(os.listdir(template_dicom_dir))
    
    sr_info_ds = dicom.read_file(series_info_source_dicom, force=True)
    
    for i, file in enumerate(sorted(os.listdir(template_dicom_dir))):
        ds=dicom.read_file(os.path.join(template_dicom_dir,file), force=True)
        if debug:
            ax1 = pylab.subplot(2,1,1)
            ax1.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    
        new_data = numpy.array(numpy.fliplr(numpy.flipud(nii_data[:,n_slices-i-1,:].T)),dtype=ds.pixel_array.dtype)
        new_data[numpy.isnan(new_data)] = 0
        new_data[new_data > 0] = 1000
        
        if overlay:
            max_bg = 10000 #TODO this should be max over all slices
            new_data += max_bg
            new_data[new_data == max_bg] = ds.pixel_array[new_data == max_bg]
        
        ds.pixel_array = new_data
        ds.PixelData = new_data.tostring()        
        
        ds.SeriesDate = sr_info_ds.SeriesDate
        ds.SeriesDescription = description
        ds.SeriesInstanceUID = sr_info_ds.SeriesInstanceUID + str(suffix)
        ds.SeriesNumber = sr_info_ds.SeriesNumber + suffix
        ds.SeriesTime = sr_info_ds.SeriesTime
        
        ds.save_as(os.path.join(output_dir, "slice%03d.dcm"%i))
        
        if debug:
            ax2 = pylab.subplot(2,1,2)
            ax2.imshow(new_data, cmap=pylab.cm.bone)
            pylab.show()
            
            
ll = [("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/3rfinger_vs_other_t_map_thr.hdr",
  "/media/data/case_studies/data/case_study_1/10",
  "/home/filo/workspace/case_studies/results/brainlab/case_1/finger",
  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
  600,"finger"),
      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/1rfoot_t_map_thr.hdr",
  "/media/data/case_studies/data/case_study_1/10",
  "/home/filo/workspace/case_studies/results/brainlab/case_1/foot",
  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
  700,"foot"),
      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/5rlips_vs_other_t_map_thr.hdr",
  "/media/data/case_studies/data/case_study_1/10",
  "/home/filo/workspace/case_studies/results/brainlab/case_1/lips",
  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
  800,"lips"),
      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_covert_verb_generation/_roi_False/_thr_method_topo_ggmm/0rtask_t_map_thr.hdr",
  "/media/data/case_studies/data/case_study_1/10",
  "/home/filo/workspace/case_studies/results/brainlab/case_1/covert_verb_generation",
  "/media/data/case_studies/data/case_study_1/5/1.2.840.113619.2.244.3596.11861638.23308.1308730975.394.dcm",
  0,"covert_verb_generation")]
            
for tr in ll:
        nii2dcm(tr[0], tr[1], tr[2], tr[3], suffix=tr[4], overlay=True, debug=False, description=tr[5])