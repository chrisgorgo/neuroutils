'''
Created on 18 Jun 2011

@author: filo
'''
import dicom
#dicom.debug()
import nibabel
import os
import numpy as np

from nipype.interfaces.base import (load_template, File, traits, isdefined,
                                    TraitedSpec, BaseInterface, Directory,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterfaceInputSpec)
from glob import glob

#nifti_file = "/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_test_subject/_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/0rfinger_t_map_thr.img"
#original_dicom_dir = "/home/filo/tmp/13"
#dicom_outdir = "/tmp/dicom_out"

#original_dicom_dir = dicom_outdir

class Nifti2DICOMInputSpec(BaseInterfaceInputSpec):
    nifti_file = File(exists=True, mandatory=True)
    template_DICOMS = traits.List(File(exists=True), mandatory=True)
    series_info_source_dicom = File(exists=True, mandatory=True)
    UID_suffix = traits.Int(mandatory=True)
    overlay = traits.Bool()
    description = traits.Str()

class Nifti2DICOMOutputSpec(TraitedSpec):
    DICOMs = traits.List(File(exists=True))
    
class Nifti2DICOM(BaseInterface):
    input_spec = Nifti2DICOMInputSpec
    output_spec = Nifti2DICOMOutputSpec
    
    def _run_interface(self, runtime):
        self.nii2dcm(self.inputs.nifti_file,
                     self.inputs.template_DICOMS,
                     os.getcwd(),
                     self.inputs.series_info_source_dicom,
                     self.inputs.UID_suffix, 
                     self.inputs.overlay,
                     False,
                     self.inputs.description)

        return runtime
    
    def _slice_filename(self, i):
        return "%s_slice%03d.dcm"%(self.inputs.description.replace(" ", "_"),i)
    
    def nii2dcm(self, nifti_file, template_dicoms, output_dir, 
                series_info_source_dicom, suffix, overlay=False, 
                debug=False, description=None):
        
        nii_data = nibabel.load(nifti_file).get_data()
        n_slices = len(template_dicoms)
        assert n_slices == nii_data.shape[1]
        
        sr_info_ds = dicom.read_file(series_info_source_dicom, force=True)
        
        dicom_handlers = [dicom.read_file(item, force=True)
                                      for item in template_dicoms]
        
        sorted_dicom_handlers = [c[1] for c in 
                                 sorted([(handler.InStackPositionNumber, 
                                          handler) 
                                         for handler in dicom_handlers])]
        max_bg = np.array([ds.pixel_array.max() for ds in sorted_dicom_handlers]).max()
            
        for i, dicom_handler in enumerate(sorted_dicom_handlers):
            ds=dicom_handler
            if debug:
                ax1 = pylab.subplot(2,1,1)
                ax1.imshow(ds.pixel_array, cmap=pylab.cm.bone)
                print ds.InStackPositionNumber
            new_data = np.array(np.fliplr(np.flipud(nii_data[:,n_slices-i-1,:].T)),dtype=ds.pixel_array.dtype)
            new_data[np.isnan(new_data)] = 0
            new_data[new_data > 0] = 1000
            
            if overlay:
                new_data += max_bg
                new_data[new_data == max_bg] = ds.pixel_array[new_data == max_bg]
            
            ds.pixel_array = new_data
            ds.PixelData = new_data.tostring()        
            
            ds.SeriesDate = sr_info_ds.SeriesDate
            ds.SeriesDescription = description
            ds.SeriesInstanceUID = sr_info_ds.SeriesInstanceUID + str(suffix)
            ds.SeriesNumber = sr_info_ds.SeriesNumber + suffix
            ds.SeriesTime = sr_info_ds.SeriesTime
            
            ds.save_as(os.path.join(output_dir, self._slice_filename(i)))
            
            if debug:
                ax2 = pylab.subplot(2,1,2)
                ax2.imshow(new_data, cmap=pylab.cm.bone)
                pylab.show()
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['DICOMs'] = [os.path.join(os.getcwd(), self._slice_filename(i)) for i in range(len(self.inputs.template_DICOMS))]
        return outputs
if __name__ == '__main__':            
    #ll = [("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/3rfinger_vs_other_t_map_thr.hdr",
    #  "/media/data/case_studies/data/case_study_1/10",
    #  "/home/filo/workspace/case_studies/results/brainlab/case_1/finger",
    #  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
    #  600,"finger"),
    #      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/1rfoot_t_map_thr.hdr",
    #  "/media/data/case_studies/data/case_study_1/10",
    #  "/home/filo/workspace/case_studies/results/brainlab/case_1/foot",
    #  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
    #  700,"foot"),
    #      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_finger_foot_lips/_roi_False/_thr_method_topo_ggmm/5rlips_vs_other_t_map_thr.hdr",
    #  "/media/data/case_studies/data/case_study_1/10",
    #  "/home/filo/workspace/case_studies/results/brainlab/case_1/lips",
    #  "/media/data/case_studies/data/case_study_1/4/1.2.840.113619.2.244.3596.11861638.23308.1308730969.874.dcm",
    #  800,"lips"),
    #      ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/1_task_name_covert_verb_generation/_roi_False/_thr_method_topo_ggmm/0rtask_t_map_thr.hdr",
    #  "/media/data/case_studies/data/case_study_1/10",
    #  "/home/filo/workspace/case_studies/results/brainlab/case_1/covert_verb_generation",
    #  "/media/data/case_studies/data/case_study_1/5/1.2.840.113619.2.244.3596.11861638.23308.1308730975.394.dcm",
    #  0,"covert_verb_generation")]
    
                
    case02 = [("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy02/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay3/rfinger_vs_other_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy02/5",
      "/home/filo/workspace/case_studies/results/brainlab/case_2/with_overlay/finger",
      "/media/data/case_studies/data/CaseStudy02/4/1.2.840.113619.2.25.4.1206130.1317397238.864.dcm",
      600,"finger"),
          ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy02/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay1/rfoot_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy02/5",
      "/home/filo/workspace/case_studies/results/brainlab/case_2/with_overlay/foot",
      "/media/data/case_studies/data/CaseStudy02/4/1.2.840.113619.2.25.4.1206130.1317397238.864.dcm",
      700,"foot"),
          ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy02/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay5/rlips_vs_other_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy02/5",
      "/home/filo/workspace/case_studies/results/brainlab/case_2/with_overlay/lips",
      "/media/data/case_studies/data/CaseStudy02/4/1.2.840.113619.2.25.4.1206130.1317397238.864.dcm",
      800,"lips")]
    
    case03 = [("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy03/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay3/rfinger_vs_other_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy03/10",
      "/home/filo/workspace/case_studies/results/brainlab/case_3/with_overlay/finger",
      "/media/data/case_studies/data/CaseStudy03/6/1.2.840.113619.2.244.3596.11861638.21263.1320070488.407.dcm",
      500,"finger"),
          ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy03/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay4/rfoot_vs_other_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy03/10",
      "/home/filo/workspace/case_studies/results/brainlab/case_3/with_overlay/foot",
      "/media/data/case_studies/data/CaseStudy03/6/1.2.840.113619.2.244.3596.11861638.21263.1320070488.407.dcm",
      600,"foot"),
          ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy03/_task_name_finger_foot_lips/_thr_method_topo_ggmm/_reslice_overlay5/rlips_vs_other_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy03/10",
      "/home/filo/workspace/case_studies/results/brainlab/case_3/with_overlay/lips",
      "/media/data/case_studies/data/CaseStudy03/6/1.2.840.113619.2.244.3596.11861638.21263.1320070488.407.dcm",
      700,"lips"),
              ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy03/_task_name_overt_word_repetition/_thr_method_topo_ggmm/_reslice_overlay0/rtask_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy03/10",
      "/home/filo/workspace/case_studies/results/brainlab/case_3/with_overlay/overt_word_repetition",
      "/media/data/case_studies/data/CaseStudy03/4/1.2.840.113619.2.244.3596.11861638.21263.1320070480.937.dcm",
      800,"overt_word_repetition"),
              ("/home/filo/workspace/case_studies/results/volumes/t_maps/thresholded/_subject_id_CaseStudy03/_task_name_covert_verb_generation/_thr_method_topo_ggmm/_reslice_overlay0/rtask_t_map_thr.hdr",
      "/media/data/case_studies/data/CaseStudy03/10",
      "/home/filo/workspace/case_studies/results/brainlab/case_3/with_overlay/covert_verb_generation",
      "/media/data/case_studies/data/CaseStudy03/5/1.2.840.113619.2.244.3596.11861638.21263.1320070483.217.dcm",
      900,"covert_verb_generation")]
                
    for tr in case03:
        nii2d = Nifti2DICOM()
        nii2d.inputs.nifti_file = tr[0]
        nii2d.inputs.template_DICOMS = glob(tr[1] + "/*.dcm")
        nii2d.inputs.series_info_source_dicom = tr[3]
        nii2d.inputs.UID_suffix = tr[4]
        nii2d.inputs.description = tr[5]
        nii2d.inputs.overlay = True
        print nii2d.run()