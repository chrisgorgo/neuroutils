'''
Created on 10 Nov 2010

@author: filo
'''

from nipype.interfaces.base import \
    BaseInterface, TraitedSpec, File, \
    traits, CommandLineInputSpec, InputMultiPath, CommandLine
from nipype.utils.misc import isdefined
from mosaic import plotMosaic
import os
from nipype.utils.filemanip import split_filename

class OverlayInputSpec(TraitedSpec):
    background = File(exists=True, mandatory=True)
    overlay = File(exists=True, mandatory=True)
    mask = File(exists=True)
    title = traits.Str("", usedefault=True)
    plane = traits.Enum("axial", "coronal", "sagital", usedefault=True)
    nrows = traits.Int(10, usedefault=True)
    bbox = traits.Bool(False, usedefault = True)
    dpi = traits.Int(300, usedefault = True)
    
    
class OverlayOutputSpec(TraitedSpec):
    plot = File(exists=True)
    

class Overlay(BaseInterface):
    input_spec = OverlayInputSpec
    output_spec = OverlayOutputSpec
    
    def _run_interface(self, runtime):
        if isdefined(self.inputs.mask):
            mask = self.inputs.mask
        else:
            mask = None

        self._plot = plotMosaic(self.inputs.background, 
           overlay=self.inputs.overlay,
           mask=mask,
           nrows=self.inputs.nrows,
           plane=self.inputs.plane,
           title=self.inputs.title,
           bbox=self.inputs.bbox,
           dpi = self.inputs.dpi)
        
        runtime.returncode=0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        return outputs
    
class PsMergeInputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr="%s", position=3, mandatory=True)
    out_file = File(argstr="-sOutputFile=%s", position=1, mandatory=True)
    settings = traits.Enum("printer", "screen", "ebook", "prepress", argstr="-dPDFSETTINGS=/%s", position=2, usedefault=True)
    
class PsMergeOutputSpec(TraitedSpec):
    merged_file = File(exists=True)
    
class PsMerge(CommandLine):
    input_spec = PsMergeInputSpec
    output_spec = PsMergeOutputSpec
    
    cmd = "gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER"
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["merged_file"] = os.path.abspath(self.inputs.out_file)
        return outputs
    
class Ps2PdfInputSpec(CommandLineInputSpec):
    ps_file = File(argstr="%s", position=1, mandatory=True, exists=True)
    settings = traits.Enum("screen", "ebook", "printer", "prepress", argstr="-dPDFSETTINGS=/%s", position=0)
    
class Ps2PdfOutputSpec(TraitedSpec):
    pdf_file = File(exists=True)
    
class Ps2Pdf(CommandLine):
    input_spec = Ps2PdfInputSpec
    output_spec = Ps2PdfOutputSpec
    cmd = "ps2pdf"
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        base, fname, ext = split_filename(self.inputs.ps_file)
        outputs["pdf_file"] = os.path.abspath(fname + ".pdf")
        return outputs
    
if __name__ == '__main__':
    plot = Overlay()
    plot.inputs.overlay = "/media/data/2010reliability/workdir/pipeline/silent_verb_generation/report/visualisethresholded_stat/_subject_id_0211541117_20101004/reslice_overlay/rthresholded_map.img"
    plot.inputs.background = "/media/data/2010reliability/E10800_CONTROL/0211541117_20101004_16864/16864/13_co_COR_3D_IR_PREP.nii"   
    plot.run()
#    from nipype.utils.filemanip import loadflat
#    a = loadflat("/media/data/2010reliability/crash-20101110-184851-filo-plot0.npz")
#    a["node"].run()