'''
Created on Feb 26, 2010

@author: filo
'''
import matplotlib as mpl
#mpl.use("PDF")
import math
import numpy as np
import nibabel as nb
import pylab as plt



axial=2
coronal=1
saggital=0
plane_dict={'axial':axial, 'coronal':coronal, 'saggital':saggital}

def boundData(data, bbox):
    return data[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    
def sliceGrid(data, nrows=7, bbox=None,plane=axial):
    if bbox:
        data = boundData(data, bbox)
    nslices = data.shape[plane]
    slice_width = getSlice(data,plane=plane).shape[0]
    slice_height = getSlice(data,plane=plane).shape[1]

    ncols = int(math.ceil(float(nslices) / nrows))

    sumimage = np.zeros((ncols * slice_width,nrows * slice_height))
    
    for j in range(ncols):
        for i in range(nrows):
            if i+j*nrows < nslices:
                sumimage[(ncols-1-j)*slice_width:(ncols-j)*slice_width,i*slice_height:(i+1)*slice_height] = getSlice(data,nslice=i+j*nrows,plane=plane)
    return sumimage

def BoundingBox(data):
    indices = np.nonzero(data)
    min_x = indices[2].min()
    min_y = indices[1].min()
    min_z = indices[0].min()
    max_x = indices[2].max()
    max_y = indices[1].max()
    max_z = indices[0].max()
    return [min_z, max_z, min_y, max_y, min_x, max_x]

def plotMosaic(background, overlay=None, mask=None,nrows=12,plane='axial', bbox=True, title="", dpi=300):
    int_plane = plane_dict[plane]
    
    bnii = nb.load(background)
    bdata = np.array(bnii.get_data(), dtype=np.float32)
    
    F = plt.figure(figsize=(8.3,11.7))
    t = F.text(0.5, 0.95, title,
               horizontalalignment='center')
    ax1 = F.add_axes([0.05, 0.25, 0.85, 0.65])
    ax1.axis("off")
    
    if overlay is not None and nb.load(overlay).get_data().max() != 0:
        onii = nb.load(overlay)
        odata = np.array(onii.get_data(), dtype=np.float32)
        if mask is not None:
            mnii = nb.load(mask)
            mdata = np.array(mnii.get_data(), dtype=np.float32)
        else:
            mdata = np.logical_not(np.logical_or(np.isnan(odata), odata == 0))
            
        if bbox:
                bbox = BoundingBox(mdata)
                odata = boundData(odata, bbox)
                mdata = boundData(mdata, bbox)
                bdata = boundData(bdata, bbox)
                          
        orig_odata = odata    
        # normalizing overlay data

        odata_not_null_min = odata[mdata != 0].min()
        odata_not_null_max = odata[mdata != 0].max()
        bdata_null_min = bdata[mdata == 0].min()
        bdata_null_max = bdata[mdata == 0].max()
        
        overlayZero = - odata_not_null_min / (odata_not_null_max - odata_not_null_min)
        if overlayZero <0 :
            overlayZero = 0
        
        mergeddata = np.ones(bdata.shape)
        mergeddata[mdata == 0] = (bdata[mdata == 0] - bdata_null_min) / (bdata_null_max - bdata_null_min)
        mergeddata[mdata != 0] = (odata[mdata != 0] - odata_not_null_min) / (odata_not_null_max - odata_not_null_min) + 1

        
        cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 1.0, 0),
                 (overlayZero*0.5+0.5, 0.0,1.0),
                 (1.0, 1.0, 0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (overlayZero*0.5+0.5, 0,0),
                   (1.0, 1.0, 0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (overlayZero*0.5+0.5, 1.0,0),
                  (1.0, 0, 0))}
        func_struct_cmap = mpl.colors.LinearSegmentedColormap('func_struct_colormap', cdict, 256)
        
        cdict = {'red': (
                 (0.0, 1.0, 0),
                 (overlayZero, 0.0,1.0),
                 (1.0, 1.0, 0)),
         'green': (
                   (0.0, 1.0, 1.0),
                   (overlayZero, 0,0),
                   (1.0, 1.0, 0)),
         'blue': (
                  (0.0, 1.0, 1.0),
                  (overlayZero, 1.0,0),
                  (1.0, 0, 0))}
        func_cmap = mpl.colors.LinearSegmentedColormap('func_colormap', cdict, 256)
        
        ax2 = F.add_axes([0.9, 0.25, 0.015, 0.65])
        ax3 = F.add_axes([0.05, 0.07, 0.9, 0.15])

        slice_grid = sliceGrid(mergeddata, nrows,plane=int_plane)
        ax1.imshow(slice_grid, cmap=func_struct_cmap, interpolation='nearest', rasterized=False, origin='lower')
         
        norma = mpl.colors.Normalize(vmin=orig_odata[mdata != 0].min(), vmax=orig_odata[mdata != 0].max())
        
        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=func_cmap,
                                           norm=norma,
                                           orientation='vertical')
        cb1.set_label('T values')
        ax3.hist(orig_odata[mdata != 0],bins=50,normed=True)
        
        del odata
        del mdata
        del orig_odata
        del mergeddata
        del slice_grid
    else:
        ax1.imshow(sliceGrid(bdata,nrows,plane=int_plane), cmap=plt.cm.gray, interpolation='nearest', rasterized=False, origin='lower')
    if title != "":
        filename = title.replace(" ", "_")+".pdf"
    else:
        filename = "plot.pdf"
        
    del bdata
    F.savefig(filename,papertype="a4",dpi=dpi)
    plt.clf()
    plt.close()
    del F
    return filename

def getSlice(data,nslice=None, plane=axial):
    if nslice is None:
        nslice =int(data.shape[plane]/2)
      
    if plane == axial:
        slice = data[:,:,nslice]     
    elif plane == coronal:
        slice = data[:,nslice,:]
    elif plane == saggital:
        slice = data[nslice,:,]
    return slice.T

def plotSlice(file,nslice=None, plane=0):
    nii = nb.load(file)
    slice = getSlice(nii.get_data(), nslice, plane)
    plt.figure() 
    plt.imshow(slice, origin="lower", interpolation='nearest', cmap=plt.cm.hot)

if __name__ == '__main__':
    #plotMosaic("/home/filo/workspace/fmri_tumour/masks/brodmann_area_4_right_d1.nii")
    #plotSlice("/home/filo/workspace/fmri_tumour/data/pilot1/10_co_COR_3D_IR_PREP.nii",plane=axial)
    #plotSlice("/home/filo/workspace/fmri_tumour/data/pilot1/10_co_COR_3D_IR_PREP.nii",plane=coronal)
    #plotSlice("/home/filo/workspace/fmri_tumour/data/pilot1/10_co_COR_3D_IR_PREP.nii",plane=saggital)
    #plotMosaic("/home/filo/workspace/fmri_tumour/data/pilot1/10_co_COR_3D_IR_PREP.nii")
    struct = "/media/data/2010reliability/E10800_CONTROL/0211541117_20101004_16864/16864/13_co_COR_3D_IR_PREP.nii"
    
    # fingertapping
    plotMosaic(struct, 
               overlay="/media/data/2010reliability/workdir/pipeline/finger_foot_lips/report/visualisethresholded_stat/_subject_id_20101014_16907/reslice_overlay/rthresholded_map.hdr",
               #mask="/media/data/2010reliability/workdir/pipeline/finger_foot_lips/_subject_id_0211541117_20101004/reslice_mask/rmask.hdr",
               nrows=10,
               plane='axial',
               title="finger tapping left brodmann area 4",
               bbox=True)
    plt.show()