
import nibabel as nb
import numpy as np
from nitime.algorithms import yule_AR_est
import nipy.neurospin.utils.design_matrix as dm
import nipy.neurospin.glm as GLM
from nipype.interfaces.base import BaseInterface, TraitedSpec, traits
from nipype.interfaces.traits import File
from nipype.utils.misc import isdefined
import os

def ARWhiten(filename):
    nii = nb.load(filename)
    data = nii.get_data()
    new_data = np.zeros(data.shape,dtype=np.float)
    
    x_size, y_size, z_size, t_size = data.shape
    
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                signal = np.array(data[x,y,z,:],dtype=np.float)              
                if signal.sum() != 0:
                    signal = signal-signal.mean()/signal.var()
                    _, ak = yule_AR_est(signal,1, None, system=True)
                    print ak
                    correction = signal*ak[0]
                    correction[1:] = correction[:-1]
                    correction[0] = correction[2]
                    new_data[x,y,z,:] = signal - correction
                    
    new_img = nb.Nifti1Image(new_data, nii.get_affine(), nii.get_header())
    nb.save(new_img, 'whitened.nii')
                
def nipy_whiten(filename, tr, hfcut):
    nii = nb.load(filename)
    timeseries = nii.get_data()
    n_scans = timeseries.shape[3]
    
    frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
    design_matrix, _ = dm.dmtx_light(frametimes, drift_model='Cosine', hfcut=hfcut)
    
    model = "ar1"
    method = "kalman"
    glm = GLM.glm()
    glm.fit(timeseries.T, design_matrix, method=method, model=model)
    explained = np.dot(design_matrix,glm.beta.reshape(glm.beta.shape[0],-1)).reshape(timeseries.T.shape).T
    residuals = timeseries - explained 
    residuals_image = nb.Nifti1Image(residuals, nii.get_affine())
    nb.save(residuals_image, 'whitened.nii')

class HighPassAndAR1WhitenInput(TraitedSpec):
    timeseries_image = File(exists=True, mandatory=True)
    TR = traits.Float(mandatory=True)
    hfcut = traits.Float(mandatory=True)
    
class HighPassAndAR1WhitenOutput(TraitedSpec):
    whitened_timeseries = File(exists=True)

class HighPassAndAR1Whiten(BaseInterface):
    input_spec = HighPassAndAR1WhitenInput
    output_spec = HighPassAndAR1WhitenOutput
    
    def _nipy_whiten(self, timeseries, tr, hfcut):
        n_scans = timeseries.shape[3]
    
        frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
        design_matrix, _ = dm.dmtx_light(frametimes, drift_model='Cosine', hfcut=hfcut)
        
        model = "ar1"
        method = "kalman"
        glm = GLM.glm()
        glm.fit(timeseries.T, design_matrix, method=method, model=model)
        explained = np.dot(design_matrix,glm.beta.reshape(glm.beta.shape[0],-1)).reshape(timeseries.T.shape).T
        residuals = timeseries - explained
        return residuals
    
    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.timeseries_image)
        timeseries = nii.get_data()
        
        whitened_data = self._nipy_whiten(timeseries, self.inputs.TR, self.inputs.hfcut)
        
        residuals_image = nb.Nifti1Image(whitened_data, nii.get_affine())
        nb.save(residuals_image, 'whitened.nii')
        
        runtime.returncode=0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["whitened_timeseries"] = os.path.abspath('whitened.nii')
        return outputs
    
nipy_whiten("/media/data/nipype_examples/spm_tutorial/workingdir/level1/_subject_id_s1/_fwhm_4/smooth/swrf3.nii", 3., 128)