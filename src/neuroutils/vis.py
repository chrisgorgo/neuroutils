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
import numpy as np
#import matplotlib as mpl
#mpl.use("Cairo")
import pylab as plt

class OverlayInputSpec(TraitedSpec):
    background = File(exists=True, mandatory=True)
    overlay = File(exists=True, mandatory=True)
    overlay_range = traits.Tuple(traits.Float, traits.Float)
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
            
        if isdefined(self.inputs.overlay_range):
            overlay_range = self.inputs.overlay_range
        else:
            overlay_range = None

        self._plot = plotMosaic(self.inputs.background, 
           overlay=self.inputs.overlay,
           overlay_range = overlay_range,
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
    
class PlotRealignemntParametersInputSpec(TraitedSpec):
    realignment_parameters = File(exists=True, mandatory=True)
    outlier_files = File(exists=True)
    title = traits.Str("Realignment parameters", usedefault=True)
    dpi = traits.Int(300, usedefault = True)
    
class PlotRealignemntParametersOutputSpec(TraitedSpec):
    plot = File(exists=True)
    
class PlotRealignemntParameters(BaseInterface):
    input_spec = PlotRealignemntParametersInputSpec
    output_spec = PlotRealignemntParametersOutputSpec
    
    def _run_interface(self, runtime):
        realignment_parameters = np.loadtxt(self.inputs.realignment_parameters)
        title = self.inputs.title
        
        F = plt.figure(figsize=(8.3,11.7))
        F.text(0.5, 0.96, self.inputs.title, horizontalalignment='center')
        ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
        handles =ax1.plot(realignment_parameters[:,0:3])
        ax1.legend(handles, ["x translation", "y translation", "z translation"], loc=0)
        ax1.set_xlabel("image #")
        ax1.set_ylabel("mm")
        ax1.set_xlim((0,realignment_parameters.shape[0]-1))
        ax1.set_ylim(bottom = realignment_parameters[:,0:3].min(), top = realignment_parameters[:,0:3].max())
        
        ax2 = plt.subplot2grid((2,2),(1,0), colspan=2)
        handles= ax2.plot(realignment_parameters[:,3:6]*180.0/np.pi)
        ax2.legend(handles, ["pitch", "roll", "yaw"], loc=0)
        ax2.set_xlabel("image #")
        ax2.set_ylabel("degrees")
        ax2.set_xlim((0,realignment_parameters.shape[0]-1))
        ax2.set_ylim(bottom=(realignment_parameters[:,3:6]*180.0/np.pi).min(), top= (realignment_parameters[:,3:6]*180.0/np.pi).max())
        
        if isdefined(self.inputs.outlier_files):
            try:
                outliers = np.loadtxt(self.inputs.outlier_files)
            except IOError as e:
                if e.args[0] == "End-of-file reached before encountering data.":
                    pass
                else:
                    raise
            else:
                ax1.vlines(outliers, ax1.get_ylim()[0], ax1.get_ylim()[1])
                ax2.vlines(outliers, ax2.get_ylim()[0], ax2.get_ylim()[1])
        
        if title != "":
            filename = title.replace(" ", "_")+".pdf"
        else:
            filename = "plot.pdf"
        
        F.savefig(filename,papertype="a4",dpi=self.inputs.dpi)
        plt.clf()
        plt.close()
        del F
        
        self._plot = filename
        
        runtime.returncode=0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["plot"] = os.path.abspath(self._plot)
        return outputs
    
class PsMergeInputSpec(CommandLineInputSpec):
    in_files = InputMultiPath(File(exists=True), argstr="%s", position=3, mandatory=True)
    out_file = File(argstr="-sOutputFile=%s", position=1, mandatory=True)
    settings = traits.Enum("prepress", "printer", "screen", "ebook",  argstr="-dPDFSETTINGS=/%s", position=2, usedefault=True)
    
class PsMergeOutputSpec(TraitedSpec):
    merged_file = File(exists=True)
    
class PsMerge(CommandLine):
    input_spec = PsMergeInputSpec
    output_spec = PsMergeOutputSpec
    
    cmd = "gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -r300"
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["merged_file"] = os.path.abspath(self.inputs.out_file)
        return outputs
    
class Ps2PdfInputSpec(CommandLineInputSpec):
    ps_file = File(argstr="%s", position=1, mandatory=True, exists=True)
    settings = traits.Enum("prepress", "screen", "ebook", "printer",  argstr="-dPDFSETTINGS=/%s", position=0)
    
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