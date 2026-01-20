# v1.03
# cone-beam tomography 
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import numpy as np
import time
import copy
import vgi
import astra
from skimage.transform import rescale, resize
__all__ = ('projfov', 'ConeRec', 'showVolume', 'createCircleMask', 'astraProjShape', 'projdist', 'multiscan', 'recmultiscan')


def createCircleMask(shape, r, center = (0, 0), smooth = True):
    h, w = shape[0:2]
    rr = r * r
    h_min, h_max = np.floor(-h // 2), np.floor(h // 2)
    w_min, w_max = np.floor(-w // 2), np.floor(w // 2)
    mask_x = np.arange(w_min, w_max) - center[0]
    mask_y = np.arange(h_min, h_max) - center[1]
    x, y = np.meshgrid(mask_x, mask_y)
    #print(y[0])
    
    if smooth:
        z = x*x / r + y*y / r
        mask = np.where(z <= r, 0.0, z - r )        
        mask = np.where(mask < 1.0, np.abs(mask - 1), 0)        
    else:
        z = x*x / rr + y*y / rr
        mask = np.where(z <= 1, 1, 0)
    mask = mask.astype(np.float32)
    return mask   

def showVolume(vol, figsize = (10, 5), normalize = False, 
    idx_d = None, idx_h = None, idx_w = None):
    D, H, W = vol.shape
    if normalize:
        vol = vgi.normalize(vol)
    if idx_d is None:
        idx_d = (0, D//4, D//2, D//4 * 3, -1)
    if idx_h is None:
        idx_h = (0, H//4, H//2, H//4 * 3, -1)
    if idx_w is None:
        idx_w = (0, W//4, W//2, W//4 * 3, -1)   
                    
    print('x-y slices')       
    imgset = vol[idx_d, :, :]
    vgi.showImageTable(imgset, 1, len(imgset), figsize=figsize)

    print('x-z slices')          
    imgset = np.swapaxes(vol[:, idx_h, :], 0, 1)
    vgi.showImageTable(imgset, 1, len(imgset), figsize=figsize) 

    print('y-z slices')             
    imgset = np.moveaxis(vol[:, :, idx_w], -1, 0)
    vgi.showImageTable(imgset, 1, len(imgset), figsize=figsize)  


# Projection FOV
# obj_radius: the radius of object.
# source_origin: the distance between the view point and the object center.
# origin_det: the distance between the object center and the detector center.
# return: 
#   fov: the fov scale factor.
def projfov(obj_radius, source_origin, origin_det):
    focal_length = source_origin + origin_det
    fov = focal_length / (source_origin - obj_radius)
    return fov

# Projection distances
# For claim detector_columns = proj_enlarge * vol_max_edge and origin_det = vol_max_edge * 0.75
# Let proj_enlarge_rate = (proj_enlarge + 1)/(proj_enlarge - 1)
# => source_origin = proj_enlarge_rate * origin_det
#
# max_edge: the maximum edge of reconstruction image
# proj_enlarge = 4: the enlargement of projection, >= 1
# od_rate = 0.75: the rate of distance between the centers of object and detector array.
def projdist(max_edge, proj_enlarge = 4, od_rate = 0.75):    
    proj_enlarge_rate = (proj_enlarge + 1)/(proj_enlarge - 1)
    origin_det = np.ceil(max_edge * od_rate)
    source_origin = np.ceil(proj_enlarge_rate * origin_det)    
    #print('source_origin, origin_det, source_det:', source_origin, origin_det, source_det)
    return source_origin, origin_det 

# Multipple stations scanning
# n_angles = 0: number of view angles.
# subvol_z_max = 16: the number of slices of each sub-volumes.
# proj_enlarge = 4: the enlargement of projection, >= 1
# od_rate = 0.75: the rate of distance between the centers of object and detector array.
# scan_range = (0, 2 * np.pi): scan range.
# det_width = 1.0, det_height = None: detector element dimension.
# meta = False： True for returning metadata.
# verbose = False: True for outputing messages.
# Return:
#   proj: projections, (n_subvols, detector_rows, n_angles, detector_columns).
#   [options]:
#   scanner, subvol_slice_set, source_origin, origin_det
# Note: n_subvols = ceil(n_slices / subvol_z_max) + 1
#       For example, n_subvols is 21 if n_slices = 314 and subvol_z_max = 16; 
#       n_subvols is 33 if n_slices = 512 and subvol_z_max = 16.
def multiscan(vol, n_angles = 0, subvol_z_max = 16, proj_enlarge = 4, od_rate = 0.75,
              scan_range = (0, 2 * np.pi), det_width = 1.0, det_height = None, 
              meta = False, verbose = False):
    vol_shape = vol.shape
    vol_slices, vol_rows, vol_columns = vol_shape
    vol_max_edge = max(vol_shape)

    # padding 
    pad_t = np.zeros([vol_slices//2, vol_rows, vol_columns])
    pad_b = np.zeros([vol_slices//2 + subvol_z_max, vol_rows, vol_columns])
    vol_pad = np.concatenate([pad_t, vol, pad_b], axis = 0)

    #subvolume
    subvol_slices = subvol_z_max * 2
    subvol_slice_set = np.arange(0, vol_slices + subvol_z_max, subvol_z_max)

    # Projection parameters
    if n_angles <= 0:    
        n_angles = int(vol_max_edge * 1.5 + 0.5)
    angles = np.linspace(scan_range[0], scan_range[1], num = n_angles, endpoint = False)
    source_origin, origin_det = projdist(vol_max_edge, proj_enlarge, od_rate)
    detector_columns = vol_max_edge * proj_enlarge
    detector_rows = subvol_slices * proj_enlarge
    proj_shape = (detector_rows, n_angles, detector_columns) 

    scanner = ConeRec(vol_shape, proj_shape, scan_range = scan_range, angles = angles,
                      det_width = det_width, det_height = det_height,
                      source_origin = source_origin, origin_det =origin_det)
    # Projection 
    proj = []
    for t, slice_s in enumerate(subvol_slice_set):
        time_s = time.time()
        slice_e = slice_s + vol_slices # not padded volume shape!
        shifted_vol = vol_pad[slice_s:slice_e]    
        proj_t = scanner.project(shifted_vol)
        proj += [proj_t]
        if verbose:
            print('multiscan::proj_%d'%t, proj_t.shape, proj_t.dtype, vgi.metric(proj_t))    
            print('multiscan::projection time:', time.time() - time_s)    
    proj = np.array(proj)  

    if meta:
        return proj, scanner, subvol_slice_set, source_origin, origin_det
    else:
        scanner.release()
        return proj
# def multiscan ............


def recmultiscan(proj, vol_shape, ds_int = 1, subvol_z_max = 16, proj_enlarge = 4, od_rate = 0.75,
              scan_range = (0, 2 * np.pi), det_width = 1.0, det_height = None, 
              anti_aliasing = False, 
              algo = 'FDK_CUDA', iterations = 1000,
              meta = False, verbose = False):
    rec_vol_shape = vol_shape
    vol_max_edge = max(vol_shape)
    n_subvol, detector_rows, n_angles, detector_columns = proj.shape
    proj_t_shape = proj.shape[1:]
    angles = np.linspace(scan_range[0], scan_range[1], num = n_angles, endpoint = False) 
    source_origin, origin_det = projdist(vol_max_edge, proj_enlarge, 0.75)
     # Downsampling setup
    if ds_int > 1:
        time_s = time.time()
        if verbose:
            print('recmultiscan::projection data downsampling')
        proj_ds_shape = (n_subvol, detector_rows // ds_int, n_angles, detector_columns // ds_int)        
        proj = resize(proj, proj_ds_shape, anti_aliasing=anti_aliasing)
        proj_t_shape = proj.shape[1:]
        proj /= ds_int
        source_origin /= ds_int
        origin_det /= ds_int
        subvol_z_max //= ds_int         
        rec_vol_shape = (rec_vol_shape[0] // ds_int, rec_vol_shape[1] // ds_int, rec_vol_shape[2] // ds_int)
        if verbose:
            print('recmultiscan::proj', proj.shape)
            print('recmultiscan::projection data downsampling time:', time.time() - time_s)  
   
    #subvolume
    rec_slices, rec_rows, rec_columns = rec_vol_shape
    subvol_slices = subvol_z_max * 2
    subvol_shape = (subvol_slices, rec_rows, rec_columns)
    if subvol_z_max > 1:        
        z_w = np.linspace(0, 1, subvol_z_max, dtype = proj.dtype)
    else:
        z_w = np.array([0.5], dtype = proj.dtype)
    z_w = np.concatenate([z_w, z_w[::-1]])

    ct = ConeRec(subvol_shape, proj_t_shape, scan_range = scan_range, angles = angles,
                 det_width = det_width, source_origin = source_origin, origin_det = origin_det,
                 algo = algo, iterations = iterations)

  
    rec = np.zeros(rec_vol_shape, dtype = proj.dtype)
    slice_s = 0
    for t, proj_t in enumerate(proj):
        time_s = time.time()
        rec_t = ct.reconstruct(proj_t)
        if verbose:
            print('recmultiscan::rec_%d'%t, rec_t.shape, rec_t.dtype)
            print('recmultiscan::reconstruction time:', time.time() - time_s)  
        #if verbose:
        #    print('recmultiscan::rec_%d'%t, rec_t.shape, rec_t.dtype, vgi.metric(rec_t))             
        rec_t = np.clip(rec_t, 0, None)   
        if t == 0:         
            #rec[slice_s:subvol_z_max] += rec_t[subvol_z_max:] * z_w[subvol_z_max:, None, None]
            rec_src = rec[slice_s:subvol_z_max]
            ze = subvol_z_max+rec_src.shape[0]
            rec[slice_s:subvol_z_max] += rec_t[subvol_z_max:ze] * z_w[subvol_z_max:ze, None, None]

        else:
            slice_e = np.clip(slice_s + subvol_slices, 0, rec_slices)
            subvol_slices_t = slice_e - slice_s            
            rec[slice_s:slice_e] += rec_t[:subvol_slices_t] * z_w[:subvol_slices_t, None, None]
            slice_s += subvol_z_max # for t > 1 

        #if verbose:
            #print('recmultiscan::reconstruction time:', time.time() - time_s)  
            #vgi.showImage(rec[:, rec_rows//2, :])
        
    if ds_int > 1:
        if verbose:
            print('recmultiscan::reconstructed volume resizing')
        rec = resize(rec, vol_shape, anti_aliasing=True)
        
    if meta:
        return rec, proj, ct, source_origin, origin_det
    else:
        ct.release()
        return rec   
# def recmultiscan ............
 
 # From (views, slices, detectors) = (slices, views, detectors)
def astraProjShape(proj):
    return np.swapaxes(proj, 0, 1)    

# ----------------------------------------------------
# Projection shapes: (detector_rows, angles, detector_columns)
class ConeRec:
    def __init__(self, vol_shape, proj_shape, scan_range = (0, 2 * np.pi), angles = None, volume = None, proj = None,
                 det_width = 1.0, det_height = None, source_origin = 640., origin_det = 384.,
                 algo = 'FDK_CUDA', iterations = 1000):
        self.vol_shape = vol_shape      # (d, h, w)
        self.depth, self.height, self.width = self.vol_shape
        self.proj_shape = proj_shape    # (slices, angles, detectors)
        self.n_det_rows, self.n_angles, self.n_det_cols = self.proj_shape
        self.scan_range = scan_range
        self.proj_mode = 'cone'      
        # create_vol_geom(Y, X, Z)``:  
        self.vol_geom = astra.create_vol_geom(self.height, self.width, self.depth)
        self.vol_id = astra.data3d.create('-vol', self.vol_geom, data = volume)
        if angles is None:
            self.angles = np.linspace(self.scan_range[0], self.scan_range[1], self.n_angles, False)
        else:
            self.angles = angles
        self.det_width = det_width
        self.det_height = self.det_width if det_height is None else det_height
        self.source_origin = source_origin
        self.origin_det = origin_det

        # create_proj_geom('cone', detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, source_det)
        self.proj_geom = astra.create_proj_geom(self.proj_mode, 
                                                self.det_width, self.det_height,
                                                self.n_det_rows, self.n_det_cols,
                                                self.angles, 
                                                self.source_origin,
                                                self.origin_det)  
        self.proj_id   = astra.data3d.create('-sino', self.proj_geom, data = proj)

        # Available algorithms:
        # 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA'
        self.algo = algo   
        self.iterations = iterations          
        self.alg_cfg = astra.astra_dict(self.algo)
        self.alg_cfg['ProjectionDataId'] = self.proj_id
        self.alg_cfg['ReconstructionDataId'] = self.vol_id
        self.alg_id = astra.algorithm.create(self.alg_cfg)  
        

    @classmethod
    def createf(cls, vol_size, n_angles = 720, algo = 'FDK_CUDA', iterations = 1000):
        ang_range = np.pi * 2       
        vol_shape = (vol_size, vol_size, vol_size)
        det_row_count = int(vol_size * 2)  
        det_col_count = int(vol_size * 2)   
        source_origin = int(vol_size * 2.05)
        origin_det = int(vol_size * 1.)
        return cls(vol_shape = vol_shape, proj_shape = (det_row_count, n_angles, det_col_count), 
                 scan_range = (0, ang_range), 
                 angles = np.linspace(0,  ang_range, num = n_angles, endpoint = False), 
                 det_width = 1.0, source_origin = source_origin, origin_det = origin_det,
                 algo = algo, iterations = iterations)

    def create512f(cls, n_angles = 720, algo = 'FDK_CUDA', iterations = 1000):
        return cls.createf(vol_size = 512, n_angles = n_angles, algo = algo, iterations = iterations)
    @classmethod
    def create256f(cls, n_angles = 360, algo = 'FDK_CUDA', iterations = 1000):
        return cls.createf(vol_size = 256, n_angles = n_angles, algo = algo, iterations = iterations)                
    @classmethod
    def create128f(cls, n_angles = 180, algo = 'FDK_CUDA', iterations = 1000):
        return cls.createf(vol_size = 128, n_angles = n_angles, algo = algo, iterations = iterations)

    def project(self, volume = None, keep_id = False):
        if not (volume is None):
            astra.data3d.store(self.vol_id, volume)  
        proj_id, proj = astra.creators.create_sino3d_gpu(self.vol_id, self.proj_geom, self.vol_geom)
        if keep_id:
            return proj_id, proj
        else:
            proj = np.array(proj) # (slices, angles, detectors)
            astra.data3d.delete(proj_id)
            return proj

    # Projection shapes: (slices, angles, detectors)
    def reconstruct(self, proj = None):
        if not(proj is None):       
            astra.data3d.store(self.proj_id, proj)       
        astra.algorithm.run(self.alg_id, self.iterations)
        rec = astra.data3d.get(self.vol_id)   
        return rec

    # Destructor
    def release(self):
        astra.data3d.delete(self.vol_id)
        astra.data3d.delete(self.proj_id)
        astra.algorithm.delete(self.alg_id) 
 