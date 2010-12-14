import os
import numpy as np

from math import log
from copy import deepcopy
from scipy import signal
from scipy.stats import distributions

from nipype.interfaces.base import traits,\
    BaseInterface, TraitedSpec, File, Bunch,\
    InterfaceResult
import nibabel as nifti

class SimulationGeneratorInputSpec(TraitedSpec):
    fwhm = traits.Float()
    volume_shape = traits.List(traits.Int(),minlen=3, maxlen=3)
    activation_shape = traits.List(traits.Int(),minlen=3, maxlen=3)
    SNR = traits.Float()
    number_of_blocks = traits.Int()
    sim_id = traits.Int()
    
class SimulationGeneratorOutputSpec(TraitedSpec):
    functional_image = File(exists=True)

class SimulationGenerator(BaseInterface):
    '''
    Simple artificial fMRI sequence generator.
    '''
    
    input_spec = SimulationGeneratorInputSpec
    output_spec = SimulationGeneratorOutputSpec
    _gaussian_kernel = None
    
    def _make_gaussian(self, size):
        if not self._gaussian_kernel:
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            z = y[:, np.newaxis]
            x0 = y0 = z0 = size // 2
            g = np.exp(-4 * log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / self.inputs.fwhm ** 2)
            self._gaussian_kernel = g / g.sum()
        return self._gaussian_kernel

    def _gen_noisy_sequence(self, pattern):
        source = np.zeros((np.shape(pattern)[0], np.shape(pattern)[1], np.shape(pattern)[2], self.inputs.number_of_blocks*2))
        g = self._make_gaussian(self.inputs.fwhm * 2)
        for i in range(self.inputs.number_of_blocks):
            source[:, :, :, i] = signal.convolve(in1 = distributions.norm.rvs(size=np.shape(pattern)) + pattern, in2 = g, mode='same');
            source[:, :, :, i+self.inputs.number_of_blocks] = signal.convolve(distributions.norm.rvs(size=np.shape(pattern)), g, mode='same');
        return (source - source.min())*100 + 1750
    
    def _gen_pattern(self):
        pattern = np.zeros(self.inputs.volume_shape)
        
        corner = [(self.inputs.volume_shape[0] - self.inputs.activation_shape[0]) / 2,
                   (self.inputs.volume_shape[1] - self.inputs.activation_shape[1]) / 2,
                   (self.inputs.volume_shape[2] - self.inputs.activation_shape[2]) / 2]
        
        pattern[corner[0]:self.inputs.activation_shape[0]+corner[0],
                corner[1]:self.inputs.activation_shape[1]+corner[1],
                corner[2]:self.inputs.activation_shape[2]+corner[2]] = self.inputs.SNR
        return pattern
    
    def run(self):
        pattern = self._gen_pattern()
        noisy_sequence = self._gen_noisy_sequence(pattern)
        nim_tmap = nifti.Nifti1Image(noisy_sequence, np.diag([1,1,1,1]))
        nifti.save(nim_tmap,"simulated_sequence.nii")
        
        runtime = Bunch(returncode=0,
                        messages=None,
                        errmessages=None)
        outputs = self.aggregate_outputs()
        return InterfaceResult(deepcopy(self), runtime, outputs=outputs)
    
    def _list_outputs(self):
        outputs = self._outputs().get()    
        outputs['functional_image'] = os.path.abspath("simulated_sequence.nii")
        return outputs