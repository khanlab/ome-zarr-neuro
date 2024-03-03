import zarr
import dask.array as da
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
from dask.diagnostics import ProgressBar





def get_vox2ras_zarr(in_zarr_path:str,level=0) -> np.array:

    #read coordinate transform from ome-zarr
    zi = zarr.open(in_zarr_path)
    attrs=zi['/'].attrs.asdict()
    multiscale=0 #first multiscale image
    transforms = attrs['multiscales'][multiscale]['datasets'][level]['coordinateTransformations']

    # 0. reorder_xfm -- changes from z,y,x to x,y,z ordering
    reorder_xfm = np.eye(4)
    reorder_xfm[:3,:3] = np.flip(reorder_xfm[:3,:3],axis=0) #reorders z-y-x to x-y-z and vice versa

    # 1. scaling_xfm (vox2ras in spim space)
    # this matches what the ome_zarr_to_nii affine has

    scaling_xfm = np.eye(4)
    scaling_xfm[0,0]=-transforms[0]['scale'][-1] #x  # 0-index in transforms is the first (and only) transform 
    scaling_xfm[1,1]=-transforms[0]['scale'][-2] #y
    scaling_xfm[2,2]=-transforms[0]['scale'][-3] #z


    return scaling_xfm @ reorder_xfm

def get_ras2vox_zarr(in_zarr_path:str,level=0) -> np.array:
    return np.linalg.inv(get_vox2ras_zarr(in_zarr_path,level=level))




def get_bounded_subregion(points: np.array, darr: da.Array):

    """ 
    Uses the extent of points, along with the shape of the dask array,
    to return a subregion of the dask array that contains the points,
    along with the grid points corresponding to the indices from the 
    uncropped dask array (to be used for interpolation).

    if the points go outside the domain of the dask array, then it is
    capped off with floor or ceil.

    points are Nx3 or Nx4 coordinates (with/without homog coord)
    darr is the floating image dask array
    """ 
    min_extent=points.min(axis=1)[:3]
    max_extent=points.max(axis=1)[:3]

    min_extent=np.floor(np.maximum(min_extent,np.zeros(min_extent.shape)))
    max_extent=np.ceil(np.minimum(max_extent,(darr.shape[-3],darr.shape[-2],darr.shape[-1])))

    subvol = darr[min_extent[0]:max_extent[0],
                          min_extent[1]:max_extent[1],
                          min_extent[2]:max_extent[2]].compute()

     # along with grid points for interpolation
    grid_points = (np.arange(min_extent[0],max_extent[0]),
                   np.arange(min_extent[1],max_extent[1]),
                   np.arange(min_extent[2],max_extent[2]))

    return (subvol,grid_points)
    

def interp_by_block(x,
                    matrix_transform,
                    flo_darr,
                    block_info=None,
                    interp_method='linear'
                    ):
    """
    main idea here is we take coordinates from the current block (ref image)
    transform them, then in that transformed space, then interpolate the 
    floating image intensities

    since the floating image is a dask array, we need to just load a 
    subset of it -- we use the bounds of the points in the transformed 
     space to define what range of the floating image we load in
    """

    arr_location = block_info[0]['array-location']
    
    xv,yv,zv=np.meshgrid(np.arange(arr_location[0][0],arr_location[0][1]),
            np.arange(arr_location[1][0],arr_location[1][1]),
            np.arange(arr_location[2][0],arr_location[2][1]),indexing='ij')

    #reshape them into a vectors (x,y,z,1) for each point, so we can matrix multiply
    xvf=xv.reshape((1,np.product(xv.shape)))
    yvf=yv.reshape((1,np.product(yv.shape)))
    zvf=zv.reshape((1,np.product(zv.shape)))
    homog=np.ones(xvf.shape)
    
    vecs=np.vstack((xvf,yvf,zvf,homog))
    
    xfm_vecs = matrix_transform @ vecs
    
    #find bounding box required for flo vol
    (grid_points,flo_vol) = get_bounded_subregion(xfm_vecs,flo_darr)    
                  
    
    #then finally interpolate those points on the template dseg volume
    interpolated = interpn(grid_points,flo_vol,
                        xfm_vecs[:3,:].T, #
                        method=interp_method,
                        bounds_error=False,
                        fill_value=0)
    
    return interpolated.reshape(x.shape)


def apply_transform(flo_ome_zarr:str,
                    ref_nii:str,
                    flo_to_ref_affine_xfm:str,
                    channel=0,
                    level=0) -> da.Array:
    """ return dask array applying transform to floating image.
        this is a lazy function, doesn't do any work until you compute() 
        on the returned dask array.

    """

    #load dask array for floating image
    flo_darr = da.from_zarr(flo_ome_zarr,component=f'/{level}')[channel,:,:,:].squeeze()

    #load reference template dseg (this will be ref space for now)
    ref_nib = nib.load(ref_nii)
    ref_darr = da.from_array(ref_nib.get_fdata())


    #the transform can be put together from flo to ref, then inverted
    # or equivalent to going from ref to flo
    
    # e.g., from flo to ref:
    # ref_ras2vox @ affine_flo_to_ref @ flo_vox2ras

    # or, from ref to flo:
    #  flo_ras2vox @ affine_ref_to_flo @ ref_vox2ras 

    #we take the latter below:

    #get affine inverse
    affine_ref_to_flo= np.linalg.inv(np.loadtxt(flo_to_ref_affine_xfm))

    ref_vox2ras = ref_nib.affine

    matrix_transform = get_ras2vox_zarr(flo_ome_zarr) @ affine_ref_to_flo @ ref_vox2ras

    
    #perform interpolation on each block in parallel
    darr_interp=da.map_blocks(interp_by_block,
                        ref_darr, dtype=np.uint16,
                        matrix_transform=matrix_transform,
                        flo_darr=flo_darr)

    return darr_interp
                    
    

