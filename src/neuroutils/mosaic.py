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

def getFlippedData(nii):
    flip_dims = np.sign(np.diag(nii.get_affine()[0:3]))
    return nii.get_data()[::flip_dims[0], ::flip_dims[1], ::flip_dims[2]]
    

def plotMosaic(background, overlay=None, mask=None,nrows=12,plane='axial', bbox=True, title="", dpi=300, overlay_range= None):
    plotEmpty = False
    int_plane = plane_dict[plane]
    
    bnii = nb.load(background)
    bdata = np.array(getFlippedData(bnii), dtype=np.float32)
    bdata[np.isnan(bdata)] = 0
    
    F = plt.figure(figsize=(8.3,11.7))
    t = F.text(0.5, 0.95, title,
               horizontalalignment='center')
    ax1 = F.add_axes([0.05, 0.25, 0.85, 0.65])
    ax1.axis("off")
    
    if overlay is not None:
        onii = nb.load(overlay)
        odata = np.array(getFlippedData(onii), dtype=np.float32)
                         
        if np.any(np.logical_not(np.logical_or(np.isnan(odata), odata == 0))):
            if mask is not None:
                mnii = nb.load(mask)
                mdata = np.array(getFlippedData(mnii), dtype=np.float32)
            else:
                mdata = np.logical_not(np.logical_or(np.isnan(odata), odata == 0))
                
            if bbox:
                    bbox = BoundingBox(mdata)
                    odata = boundData(odata, bbox)
                    mdata = boundData(mdata, bbox)
                    bdata = boundData(bdata, bbox)
                              
            orig_odata = odata    
            # normalizing overlay data
            if overlay_range:
                odata_not_null_min = overlay_range[0]
                odata_not_null_max = overlay_range[1]
            else:
                odata_not_null_min = odata[mdata != 0].min()
                odata_not_null_max = odata[mdata != 0].max()
                if odata_not_null_min < 0:
                    if abs(odata_not_null_min) < abs(odata_not_null_max):
                        odata_not_null_min = -odata_not_null_max
                    else:
                        odata_not_null_max = - odata_not_null_min
                
            if odata_not_null_min == odata_not_null_max:
                odata_not_null_min = 0
                
            overlayZero = - odata_not_null_min / (odata_not_null_max - odata_not_null_min)
            if overlayZero <0 :
                overlayZero = 0
                
            bdata_null_min = bdata[mdata == 0].min()
            bdata_null_max = bdata[mdata == 0].max()
            
    
            
            mergeddata = np.ones(bdata.shape)
            mergeddata[mdata == 0] = (bdata[mdata == 0] - bdata_null_min) / (bdata_null_max - bdata_null_min)
            mergeddata[mdata != 0] = (odata[mdata != 0] - odata_not_null_min) / (odata_not_null_max - odata_not_null_min) + 3
    
            
            cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.25, 1.0, 1.0),
                     (0.75, 1.0,0),
                     (0.75 + 0.25*overlayZero, 0.0,1.0),
                     (1.0, 1.0, 0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.75, 1.0, 1.0),
                       (0.75 + 0.25*overlayZero, 0,0),
                       (1.0, 1.0, 0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.25, 1.0, 1.0),
                      (0.75, 1.0,1.0),
                      (0.75 + 0.25*overlayZero, 1.0,0),
                      (1.0, 0, 0))}
            func_struct_cmap = mpl.colors.LinearSegmentedColormap('func_struct_colormap', cdict, 1024)
            
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
            func_cmap = mpl.colors.LinearSegmentedColormap('func_colormap', cdict, 1024)
            
            ax2 = F.add_axes([0.9, 0.25, 0.015, 0.65])
            ax3 = F.add_axes([0.05, 0.07, 0.9, 0.15])
    
            slice_grid = sliceGrid(mergeddata, nrows,plane=int_plane)
            ax1.imshow(slice_grid, cmap=func_struct_cmap, interpolation='nearest', rasterized=False, origin='lower', 
                       norm = mpl.colors.Normalize(vmin=0, vmax=4))
            
            if overlay_range:
                norma = mpl.colors.Normalize(vmin=overlay_range[0], vmax=overlay_range[1])
            else:
                norma = mpl.colors.Normalize(vmin=odata_not_null_min, vmax=odata_not_null_max)
            #norma = mpl.colors.Normalize(vmin=mergeddata.min(), vmax=mergeddata.max())
            
            
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
            if overlay_range:
                ax3.set_xlim(left = overlay_range[0], right = overlay_range[1])
            
            del odata
            del mdata
            del orig_odata
            del mergeddata
            del slice_grid
        else:
            plotEmpty = True
    else:
        plotEmpty = True
        
    if plotEmpty:
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
    #struct = "/home/filo/workspace/2010reliability/results/volumes/T1/_subject_id_08143633-aec2-49a9-81cf-45867827b871/wmr13_co_COR_3D_IR_PREP_maths.nii"
    struct = "/home/filo/workspace/2010reliability/results/volumes/T1/_subject_id_3a3e1a6f-dc92-412c-870a-74e4f4e85ddb/wmr14_co_COR_3D_IR_PREP_maths.nii"
    # fingertapping
    plotMosaic(struct, 
               overlay="/home/filo/workspace/2010reliability/results/volumes/t_maps/thresholded/_subject_id_08143633-aec2-49a9-81cf-45867827b871/_session_first/_task_name_finger_foot_lips/_thr_method_topo_fdr/0rfinger_t_map_thr.img",
               #mask="/media/data/2010reliability/workdir/pipeline/finger_foot_lips/_subject_id_0211541117_20101004/reslice_mask/rmask.hdr",
               nrows=10,
               plane='axial',
               title="bla",
               bbox=False,
               #overlay_range=(0, 14)
               )
    plt.show()
