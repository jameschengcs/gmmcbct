# V1.008
# errormap

import os
import numpy as np
import sys
sys.path.append('../')
import vgi
import json
import astra
import torch 
import time
import cc3d
from vgi.ct import ConeRec, astraProjShape
from vgi.ssim import SSIM
from operator import itemgetter
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

gpu_dtype = torch.float 
cpu_dtype = np.float32
n_parameters = 11

# Get indices of subvolumes with a fixed interval d.
start_idx = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                     dtype = np.int32)

# No binarization if bin_threshold is None
def errorMap(vol, target_proj, scanner, clip_min = 0.0, clip_max = None, sigma = 1.0, bin_threshold = 0.5, meta = False):
    reproj = scanner.project(vol)    
    proj_d = target_proj - reproj    
    proj_d_abs = np.abs(proj_d)   
    recon = scanner.reconstruct(proj_d_abs)
    
    err_map = np.clip(recon, 0.0, None)
    err_map = vgi.normalize(err_map)
    err_map = gaussian_filter(err_map, sigma=sigma)   
    err_map = vgi.normalize(err_map)
    if not (bin_threshold is None):
        err_map = np.where(err_map > bin_threshold, 1.0, 0.0)    
    if meta:
        return err_map, reproj, proj_d, recon
    else:
        return err_map
    
def bbsize(bb):
    n = 1
    shape = []
    for slc in bb:
        d = slc.stop - slc.start
        n *= d
        shape += [d]
    return n, shape
def bbstr(bb):
    s = '('
    for slc in bb:
        s += str(slc.start) + ":" + str(slc.stop) + ","
    s += ')'
    return s   

def shrinkbb(bb, s):
    bb_out = tuple()
    for slc in bb:
        mid = int(slc.start + slc.stop) // 2
        r = int(slc.stop - slc.start)
        w = r // 2        
        ws = int(w * s)
        add = ws & 1 if ws > 0 else 1
        slc_i = slice(mid - ws, mid + ws + add, slc.step)
        bb_out += (slc_i,)
    return bb_out



    
def errorStrokes(vol, err_map, min_vx = 7):
    err_cc = cc3d.connected_components(err_map)     
    stats = cc3d.statistics(err_cc)
    voxel_counts = stats['voxel_counts']
    bounding_boxes = stats['bounding_boxes']
    centroids = stats['centroids']
    n = voxel_counts.size
    # noise removing
    strokes = []
    err_stk = np.array(err_map)
    for i in range(1, n):
        bb = bounding_boxes[i]  
        if voxel_counts[i] >= min_vx:            
            mean_vx = np.mean(vol[bb])
            size, shape = bbsize(bb)
            center = centroids[i]
            stroke = {'size':size, 'shape':shape, 'bb':bb, 'center':center, 'mean_vx':mean_vx  }
            strokes += [stroke]
        else:
            err_stk[bb] = 0
    strokes.sort(key = itemgetter('size'), reverse = True)
    return strokes, err_stk

def errorGaussians(vol, data_min_p, target_proj, scanner, 
                   min_init_size = 5.0,
                   clip_min = 0.0, clip_max = None, sigma = 1.0,
                   bin_threshold = 0.5, min_vx = 7,
                   rep = 2, rep_vx = 27, rep_s = 0.5 ):
    err_map = errorMap(vol, target_proj, scanner, 
                       clip_min = clip_min, clip_max = clip_max, 
                       sigma = sigma, bin_threshold = bin_threshold)
    strokes, err_stk = errorStrokes(vol, err_map, min_vx = min_vx)    
    n_vx = vol.size
    G = None
    for i, stroke in enumerate(strokes):        
        rep_i = max(rep, 1) if stroke['size'] >= rep_vx else 1
        rep_si = 1.0
        bb_i = stroke['bb']
        for j in range(rep_i):
            ge_center = stroke['center'] + data_min_p
            uz, uy, ux = ge_center
            ge_size = np.array(stroke['shape']) / 2
            ge_size *= rep_si # for repreating
            sz, sy, sx = ge_size
            min_s = min([sz, sy, sx])
            if min_s < min_init_size:
                sr = min_init_size / min_s
                sz *= sr
                sy *= sr
                sx *= sr
            #print('errorGaussians:', i, stroke['size'], sx, sy, sz)
            alpha = 0.9#1.0 - stroke['size'] / n_vx
            beta = np.mean(vol[bb_i])#stroke['mean_vx']   np.mean(vol[bb])
            if np.isnan(beta):
                print('beta nan:', bb_i, stroke['bb'])
            # [ux, uy, uz, sx, sy, sz, rx, ry, rz, alpha, beta]
            g = np.array([[ux, uy, uz, sx, sy, sz, 0.0, 0.0, 0.0, alpha, beta]]) # (1, #par)
            if G is None:
                G = g
            else:
                G = np.concatenate([G, g])
            rep_si *= rep_s
            bb_i = shrinkbb(bb_i, rep_s)
        # repeating:
        



    return G

def subvolID(shape, start_i = 0, d = 2):
    idx_s = start_idx[start_i]
    idx = None
    for z in range(idx_s[0], shape[0], d):
        for y in range(idx_s[1], shape[1], d):
            for x in range(idx_s[2], shape[2], d):
                idx_i = np.array([z, y, x], dtype = np.int32)
                idx_i = np.expand_dims(idx_i, 0)
                if idx is None:
                    idx = idx_i
                else:
                    idx = np.concatenate([idx, idx_i])
    return idx

# Uniformly dividing a volume space 
# box :[2, 3]
# min_size: [3]
# outputs:
#   B: boxes:  [levels][slices, rows, columns, 2, 3]
#   C: shapes: [levles][slices, rows, columns]
#   idx: indeices: [levles][offests][boxes, 3]
def divideBox(box, min_size = np.array([8., 8., 8.]), d = 2):
    box_min = box[0].astype(np.float) 
    box_max = box[1].astype(np.float)
    box_size_i = box_max - box_min
    B = []
    C = []
    idx = []
    while box_size_i[0] >= min_size [0] and box_size_i[1] >= min_size [1] and box_size_i[2] >= min_size [2]:
        #print('box_size_i', box_size_i)
        B_lv = None
        box_min_i0 = box_min
        box_max_i0 = box_min_i0 + box_size_i
        box_min_i = np.array(box_min_i0)
        box_max_i = np.array(box_max_i0)
        nz = 0
        while box_max_i[0] <= box_max[0]:
            box_min_i[1] = box_min_i0[1]
            box_max_i[1] = box_max_i0[1]
            ny = 0
            B_slice = None
            while box_max_i[1] <= box_max[1]:
                box_min_i[2] = box_min_i0[2]
                box_max_i[2] = box_max_i0[2]
                nx = 0
                B_row = None
                while box_max_i[2] <= box_max[2]:
                    #print('box i', box_min_i, box_max_i)
                    box = np.array(np.stack([box_min_i, box_max_i]))
                    box = np.expand_dims(box, 0)
                    if B_row is None:
                        B_row = box
                    else:
                        B_row = np.concatenate([B_row, box])
                    box_min_i[2] = box_max_i[2]
                    box_max_i[2] += box_size_i[2]
                    nx += 1
                # x loop
                B_row = np.expand_dims(B_row, 0)
                if B_slice is None:
                    B_slice = B_row
                else:
                    B_slice = np.concatenate([B_slice, B_row])
                box_min_i[1] = box_max_i[1]
                box_max_i[1] += box_size_i[1]   
                ny += 1
            # y loop
            box_min_i[0] = box_max_i[0]
            box_max_i[0] += box_size_i[0]
            nz += 1
            B_slice = np.expand_dims(B_slice, 0)
            if B_lv is None:
                B_lv = B_slice
            else:
                B_lv = np.concatenate([B_lv, B_slice])  
        # z loop
        B += [B_lv]
        C += [np.array([nz, ny, nx])]
        box_size_i /= 2
    # level loop
    for i_lv, C_lv in enumerate(C):
        idx_lv = []
        for start_i in range(8):
            idx_i = subvolID(C_lv, start_i, d)  
            if not(idx_i is None):
                idx_lv += [idx_i]
        idx += [idx_lv]    
    return B, C, idx

def midImages(_volume, mid = None):
    if torch.is_tensor(_volume):        
        vol = vgi.toNumpy(_volume)
    else:
        vol = _volume
    if mid is None:
        mid = [vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2]
    return vol[mid[0]], vol[:, mid[1], :], vol[:, :, mid[2]]

def show(_volume, mid = None, mip = False, figsize = None):
    if torch.is_tensor(_volume):        
        vol = vgi.toNumpy(_volume)
    else:
        vol = _volume
    if mid is None:
        mid = [vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2]
    vgi.showImage(vol[mid[0]], figsize = figsize)
    vgi.showImage(vol[:, mid[1], :], figsize = figsize)
    vgi.showImage(vol[:, :, mid[2]], figsize = figsize)
    if mip:
        vgi.showImage(np.max(vol, axis =0), figsize = figsize)
        vgi.showImage(np.max(vol, axis =1), figsize = figsize)
        vgi.showImage(np.max(vol, axis =2), figsize = figsize)    

def intensity_adjust(data, gt_data):
    data_adj = data
    #data_adj = np.clip(data, 0.0, data.max())
    #data_adj = vgi.normalize(data)
    #data_adj = vgi.normalize(data_adj, 0.0, gt.max())
    data_adj = np.clip(data_adj, 0.0, data_adj.max())
    ua = np.mean(data_adj)
    ub = np.mean(gt_data)
    data_adj = data_adj * (ub / ua)
    #data_adj = np.clip(data_adj, 0.0, data_adj.max())
    return data_adj
    
def evaluate(data, gt_data, name = '', adjust = True):
    if adjust:
        data = intensity_adjust(data, gt_data)
    diff = data - gt_data
    v_mae = np.mean(np.abs(diff))
    v_mse = mse(data, gt_data) 
    v_ssim = ssim(data, gt_data, data_range = gt_data.max() - gt_data.min())
    v_psnr = psnr(data, gt_data, data_range = gt_data.max() - gt_data.min())
    #print('mae', 'mse', 'ssim', 'psnr')
    #print(name, v_mae, v_mse, v_ssim, v_psnr)
    return v_mae, v_mse, v_ssim, v_psnr

def strev(ev):
    return '%0.4f %0.6f %0.4f %0.2f'%ev

def subvol(vol, bd):
    return vol[..., bd[0, 0]:bd[1, 0], bd[0, 1]:bd[1, 1], bd[0, 2]:bd[1, 2]]
    
# deafult data transform:
def notrans(_data):
    return _data

# .................................................
# CBCT transform
# Input projection shapes: (slices, angles, detectors)
# Output projection shapes: (angles, slices, detectors)
class CTtransform(ConeRec):    # class designing
    def __init__(self, vol_shape, proj_shape, scan_range = (0, 2 * np.pi), angles = None, volume = None, proj = None,
                 det_width = 1.0, source_origin = 512., origin_det = 512.,
                 algo = 'FDK_CUDA', iterations = 1000):
        super().__init__(vol_shape = vol_shape, proj_shape = proj_shape, scan_range = scan_range, 
                         angles = angles, volume = volume, proj = proj, 
                         det_width = det_width, source_origin = source_origin, origin_det = origin_det,
                         algo = algo, iterations = iterations)

    def __call__(self, _data):
        data = vgi.toNumpy(_data)        
        proj = self.project(data) # (slices, angles, detectors) 
        _proj = torch.tensor(proj, dtype = _data.dtype, device = _data.device)
        _proj = _proj.swapaxes(0, 1) # (angles, slices, detectors) 
        return _proj

# .................................................
# Gaussian ellipsoid reconstruction 3D
# g = [[ux], [uy], [uz], [sx], [sy], [sz], [rx], [ry], [rz], [alpha], [beta]]
# |g| = 11
class Composer:
    def tensor(self, data):
        return torch.tensor(data, dtype = gpu_dtype, device = self.device).to(non_blocking = True)

    def zeros(self, shape):
        return torch.zeros(shape, dtype = gpu_dtype, device = self.device).to(non_blocking = True)
    def ones(self, shape):
        return torch.ones(shape, dtype = gpu_dtype, device = self.device).to(non_blocking = True)  
    def full(self, shape, value):
        return torch.full(shape, value, dtype = gpu_dtype, device = self.device).to(non_blocking = True)
    def toTorchImage(self, image):
        return vgi.toTorchImage(image = image, dtype = gpu_dtype, device = self.device).to(non_blocking = True)
    
    def __init__(self, shape, target = None,  
                 ssim_window = 7, g_min = None, g_max = None,   
                 seed = 8051, volume_batch_size = 4, slice_batch_size = 128, 
                 max_batch_size = 1, loss = 'SSIML1B', loss_axis = 1,
                 SSIM_weight = 0.8, 
                 data_trans = notrans, 
                 gpu = True):
        self.gpu = gpu
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")    
        self.shape = shape    
        self._volume = self.zeros(self.shape)

        self.target = target        
        self.target_shape = None
        self.target_shape_sum = 0
        self.val_range = 1.0
        self.max_batch_size = max_batch_size # The maximum batch size of ellipsoids
        self.volume_batch_size = volume_batch_size
        self.slice_batch_size = slice_batch_size
        self.batch_idx_z = None
        self.batch_idx_y = None
        self.batch_idx_x = None
        if not self.target is None:
            self._target = self.tensor(target) # (slices, height, width)
            self.val_range = self._target.max() - self._target.min()
            self.target_shape = self.target.shape
            self.target_shape_sum = self.target_shape[0] + self.target_shape[1] + self.target_shape[2]
            self.batch_idx_z = vgi.batchIdx(self.target_shape[0], self.slice_batch_size)
            self.batch_idx_y = vgi.batchIdx(self.target_shape[1], self.slice_batch_size)
            self.batch_idx_x = vgi.batchIdx(self.target_shape[2], self.slice_batch_size)  
            
        # Constants            
        self._one = self.tensor(1.0)    
        self._zero = self.tensor(0.0)             
        self.pi2 = np.pi + np.pi
        self.pih = np.pi / 2
        self.piq = np.pi / 4          
        self.seed = seed
        self._expThres = self.tensor(-15.0)
        self.ssim_window = ssim_window
        
        # Geometry
        shape_array = np.array(self.shape)
        self.max_i = shape_array - 1
        self.min_p = -(shape_array - 1) // 2
        self.max_p = (shape_array + 1) // 2
        self.box = np.array([self.min_p, self.max_p])
        self.data_min_p = -shape_array // 2       
        self.data_max_p = shape_array + self.data_min_p  
        self._min_p = self.tensor(self.min_p)
        self._max_p = self.tensor(self.max_p)
        self._data_min_p = self.tensor(self.data_min_p)
        self._data_max_p = self.tensor(self.data_max_p)     
        
        self._Z, self._Y, self._X = torch.meshgrid(
                                          torch.arange(self.data_min_p[0], self.data_max_p[0], dtype = gpu_dtype, device = self.device), 
                                          torch.arange(self.data_min_p[1], self.data_max_p[1], dtype = gpu_dtype, device = self.device),
                                          torch.arange(self.data_min_p[2], self.data_max_p[2], dtype = gpu_dtype, device = self.device),                                          
                                          indexing = 'ij')
        self._P = torch.stack([self._X, self._Y, self._Z, self.ones(self._X.shape)])
        
        # g = [ux, uy, uz, sx, sy, sz, rx, ry, rz, alpha, beta]
        # |g| = 11
        if g_min is None:
            self.g_min = np.array([self.min_p[2], self.min_p[1], self.min_p[0],
                                   1.0, 1.0, 1.0, 
                                   -self.piq, -self.piq, -self.piq,
                                   0.01, 0.0])
        else:
            self.g_min = np.array(g_min)
        if g_max is None:
            self.g_max = np.array([self.max_p[2], self.max_p[1], self.max_p[0],
                                   self.max_p[2], self.max_p[1], self.max_p[0],
                                   self.piq, self.piq, self.piq,
                                   0.9, 1.0])
        else:
            self.g_max = np.array(g_max)

        self._g_min = self.tensor(self.g_min)
        self._g_max = self.tensor(self.g_max)
        angle_step = np.pi / 180.
        voxel_step = 1/256
        self._step_size = self.tensor( [0.5, 0.5, 0.5,
                                        0.5, 0.5, 0.5,
                                        angle_step, angle_step, angle_step, 
                                        voxel_step, voxel_step]) #[parameters]
 
        self._G = None  
        self.area_f = 3.0 
        self._mat_I = torch.eye(4, dtype = gpu_dtype, device = self.device)

        self.data_trans = data_trans  

        # Loss

        self.loss = loss
        self.loss_axis = loss_axis
        self.SSIM_weight = SSIM_weight
        self._mse = torch.nn.MSELoss()
        self._L1 = torch.nn.L1Loss()        
        self._SSIM = SSIM(val_range = self.val_range, window_size = self.ssim_window, mean_axis = (1, 2, 3))
        self.lf = self.getLoss(self.loss)
    # Composer::__init__
        
    def volumeLocation(self, p):
        return np.clip(p - self.data_min_p, [0., 0., 0.], self.max_i).astype(np.int32)

    def subbox(self, center, size):
        box_min = center - size
        box_max = center + size + 1
        box_min = np.clip(box_min, [0, 0, 0], self.max_i).astype(np.int32)
        box_max = np.clip(box_max, [0, 0, 0], self.max_i).astype(np.int32)
        return box_min, box_max

    def betaSampling(self, g, volume, box_size):
        p = g[2::-1]
        vol_p = self.volumeLocation(p)
        box_min, box_max = self.subbox(vol_p, box_size)
        beta = np.mean(volume[box_min[0]:box_max[0], box_min[1]:box_max[1], box_min[2]:box_max[2]])                
        return beta

    def setVolume(self, _volume):
        self._volume = _volume

    def setMinSize(self, sx, sy = None, sz = None):
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx

        self.g_min[3] = sx
        self.g_min[4] = sy
        self.g_min[5] = sz
        self._g_min = self.tensor(self.g_min)
        

    # Creating a range for initializing g
    def rangeInitG(self, loc_rate = 0.85, size_range = None, angle_range = None, alpha_range = None, beta_range = None):
        g_init_min = np.array(self.g_min)
        g_init_max = np.array(self.g_max)
        g_init_min[0:3] *= loc_rate
        g_init_max[0:3] *= loc_rate

        def setRange(r, i, n):
            if not(r is None):
                j = i + n
                if np.isscalar(r):
                    g_init_min[i:j] = r
                    g_init_max[i:j] = r
                else:
                    g_init_min[i:j] = r[0]
                    g_init_max[i:j] = r[1]            

        setRange(size_range, 3, 3)
        setRange(angle_range, 6, 3)
        setRange(alpha_range, 9, 1)
        setRange(beta_range, 10, 1)
           
        g_init_min = np.clip(g_init_min, self.g_min, self.g_max)
        g_init_max = np.clip(g_init_max, self.g_min, self.g_max)
        return g_init_min, g_init_max
    # Composer::rangeInitG

    #_I:(volumns, slices, height, width)
    # Each loss function only computes the error of single data item rather than batch.
    def lossL1(self, _I):
        _loss = (_I -  self._target).abs()
        _loss = _loss.mean(dim = (-1, -2, -3))
        return _loss                 
    
    def lossMSE(self, _I):
        _loss = ((_I -  self._target) ** 2.0)
        _loss = _loss.mean(dim = (-1, -2, -3))
        _loss = _loss ** 0.5
        return _loss  
    
    

    # _I: (volumes, depth, height, width)
    def lossSSIM(self, _I):
        if len(_I.shape) == 3:
            _I = _I.unsqueeze(0)
        self._SSIM.mean_axis = (-1, -2, -3)
        def core(_img1, _img2, _sum):
            n_volumes = _img1.shape[0]
            n_slices = _img1.shape[1]
            _img1 = _img1.flatten(0, 1).unsqueeze(1)
            _img2 = _img2.tile((n_volumes, 1, 1))
            _img2 = _img2.unsqueeze(1)
            #print('_img1', _img1.shape)
            #print('_img2', _img2.shape)
            _ssim = self._SSIM.forward(_img1, _img2)
            _ssim = _ssim.reshape([n_volumes, n_slices])
            _ssim = _ssim.sum(dim = -1)
            if _sum is None:
                _sum = _ssim
            else:
                _sum += _ssim   
            return _sum         

        _ssim_sum = None
        n_batches = 0
        for idx in self.batch_idx_z:
            i, j = idx
            _img1 = _I[:, i:j]
            _img2 = self._target[i:j]
            _ssim_sum = core(_img1, _img2, _ssim_sum)
            n_batches += 1
    
        _ssim_mean = _ssim_sum / self._target.shape[0]
        _loss = self._one - _ssim_mean        
        return _loss  
    # Composer::lossSSIM 

    def lossSSIML1(self, _I):
        if len(_I.shape) == 3:
            _I = _I.unsqueeze(0)     
        #print('lossSSIML1::_I', _I.shape)   
        #print('lossSSIML1::self._target', self._target.shape)   
        self._SSIM.L1 = True
        self._SSIM.alpha = self.SSIM_weight
        self._SSIM.mean_axis = (-1, -2, -3)        
        def core(_img1, _img2, _sum):
            n_volumes = _img1.shape[0]
            n_slices = _img1.shape[1]
            _img1 = _img1.flatten(0, 1).unsqueeze(1)
            _img2 = _img2.tile((n_volumes, 1, 1))
            _img2 = _img2.unsqueeze(1)
            _ssim = self._SSIM.alpha - self._SSIM.forward(_img1, _img2)
            _ssim = _ssim.reshape([n_volumes, n_slices])
            _ssim = _ssim.sum(dim = -1)
            if _sum is None:
                _sum = _ssim
            else:
                _sum += _ssim   
            return _sum         

        _ssim_sum = None
        n_batches = 0
        for idx in self.batch_idx_z:
            i, j = idx
            _img1 = _I[:, i:j]
            _img2 = self._target[i:j]
            _ssim_sum = core(_img1, _img2, _ssim_sum)
            n_batches += 1
        _loss = _ssim_sum / self._target.shape[0]   
        return _loss 
    # Composer::lossSSIML1               
    
    def getLoss(self, loss = 'ssim'):
        lf = None
        if loss == 'ssim' or loss == 'SSIM':
            lf = self.lossSSIM
        elif loss == 'mse' or loss == 'MSE':   
            lf = self.lossMSE
        elif loss == 'L1':   
            lf = self.lossL1
        elif loss == 'ssimL1' or loss == 'SSIML1':
            lf = self.lossSSIML1 
        return lf     
    # Composer::getLoss

    # _g: a set of argument vectors of primitives, (n, parameters)    
    def drawGaussian(self, _G, weight_only = False, f_only = False): 
        # _G = (n, parameters)
        if not torch.is_tensor(_G):
            _G = self.tensor(_G)
        n = _G.shape[0] # the number of Gaussian ellipsoids
        _u = _G[:, 0:3]
        _s = _G[:, 3:6]
        _rx = _G[:, 6]#tensor([np.pi * 0.0, np.pi * -0.25])
        _ry = _G[:, 7]#tensor([np.pi * 0.5, np.pi * 0.0])
        _rz = _G[:, 8]#tensor([np.pi * 0.0, np.pi * 0.5])
        _alpha = _G[:, 9]
        _beta  = _G[:, 10]
         
        _ss = _s * _s
        _ss = _ss.unsqueeze(-1) # (n, 3, 1)
        _2ss = _ss + _ss

        # Transformation matrices (n, 4, 4)
        _cosrx = torch.cos(_rx)
        _sinrx = torch.sin(_rx)    
        _cosry = torch.cos(_ry)
        _sinry = torch.sin(_ry)    
        _cosrz = torch.cos(_rz)
        _sinrz = torch.sin(_rz)    
        _mat_rx = self.zeros([n, 4, 4])
        _mat_ry = self.zeros([n, 4, 4])
        _mat_rz = self.zeros([n, 4, 4])

        _mat_rx[:, 0, 0] = 1.0
        _mat_rx[:, 1, 1] =  _cosrx
        _mat_rx[:, 1, 2] = -_sinrx
        _mat_rx[:, 2, 1] =  _sinrx
        _mat_rx[:, 2, 2] =  _cosrx
        _mat_rx[:, 3, 3] = 1.0

        _mat_ry[:, 1, 1] = 1.0
        _mat_ry[:, 0, 0] =  _cosry
        _mat_ry[:, 0, 2] =  _sinry
        _mat_ry[:, 2, 0] = -_sinry
        _mat_ry[:, 2, 2] =  _cosry
        _mat_ry[:, 3, 3] = 1.0

        _mat_rz[:, 2, 2] = 1.0
        _mat_rz[:, 0, 0] =  _cosrz
        _mat_rz[:, 0, 1] = -_sinrz
        _mat_rz[:, 1, 0] =  _sinrz
        _mat_rz[:, 1, 1] =  _cosrz
        _mat_rz[:, 3, 3] = 1.0
        _mat_r = torch.matmul(_mat_rz, torch.matmul(_mat_ry, _mat_rx))
        _mat_r3 = _mat_r[:, :3, :3]
        _mat_rr = _mat_r3 * _mat_r3  
        _mat_rr = _mat_rr.transpose(-1, -2)

        _mat_t = torch.eye(4, dtype = gpu_dtype, device = self.device).tile((n, 1, 1))
        _mat_t[:, 0:3, 3] = _u

        _mat_trans =  torch.matmul(_mat_r, _mat_t)
        
        # AABB computing
        _v = torch.matmul(_mat_rr, _ss).sqrt().squeeze(-1) * self.area_f
        _u_rev = -_u
        _bd_min = (_u_rev - _v).floor().flip(-1) # flip:(x, y, z) => (depth, height, width)
        _bd_min = torch.clamp(_bd_min, self._data_min_p, self._max_p)
        _bd_min -= self._data_min_p # shift to data index
        _bd_min = _bd_min.int()
        _bd_max = (_u_rev + _v + 1).ceil().flip(-1)
        _bd_max = torch.clamp(_bd_max, self._data_min_p, self._data_max_p)
        _bd_max -= self._data_min_p
        _bd_max = _bd_max.int()      
        
        ret = None
        # out_shape = (n, 1, self.rec_shape[0], self.rec_shape[1])
        for k in range(n):            
            _bdk = torch.stack([_bd_min[k], _bd_max[k]])
            _Pk = subvol(self._P, _bdk)
            shape_k = _Pk[0].shape
            _Pk = _Pk.flatten(start_dim = 1)            
            _Ptk = torch.matmul(_mat_trans[k], _Pk)[0:3] # removing the 4th dimension (all-ones)
            _PtPtk = _Ptk * _Ptk
            #print('_PtPtk', _PtPtk.shape)
            _wk = -( (_PtPtk / _2ss[k]).sum(dim = 0) ) # -sum_xyz((3, voxels) / (3, 1)) ==> (voxels)
            #print('_wk', _wk.shape)
            _gwk = torch.where(_wk > self._expThres, torch.exp(_wk), self._zero).reshape(shape_k)  # (d, h, w)            
            #_gwk = torch.exp(_wk).reshape(shape_k)  # (d, h, w)            
            
            if weight_only:
                ret_k = _gwk
            else:
                _gak = _gwk * _alpha[k]
                _fk = (self._one - _gak) # Transparency
                if f_only:                
                    ret_k = _fk  
                else:          
                    _gbk = (_gak * _beta[k])
                    ret_k = _gbk, _fk, _gwk, _bdk            
            ret_k = [ret_k]
            if ret is None:
                ret = ret_k
            else:
                ret += ret_k
        return ret 
    # Composer::drawGaussian       

    def blendForeground(self, _volume, _foreground):
        _vol_fg, _fk = _foreground
        if _volume is None:
            _vol = _vol_fg 
        else:
            _vol = _vol_fg + _fk * _volume
        return _vol


    def blend(self, G_volumes, _volume = None, _foreground = None, blend_all = True):
        if _volume is None:
            _vol = vgi.clone(self._volume)
        else:
            _vol = vgi.clone(_volume)
        if blend_all:            
            for gvol_k in G_volumes:
                _gbk, _fk, _gwk, _bdk = gvol_k
                #print('_bdk', vgi.toNumpy(_bdk))
                _vol[ _bdk[0, 0]:_bdk[1, 0], _bdk[0, 1]:_bdk[1, 1], _bdk[0, 2]:_bdk[1, 2]] = _gbk + _fk * subvol(_vol, _bdk)
            if not(_foreground is None):
                _vol = self.blendForeground(_vol, _foreground)
            return _vol
        else:
            _volumes = None            
            for gvol_k in G_volumes:
                _gbk, _fk, _gwk, _bdk = gvol_k
                _volume_k = vgi.clone(_volume)
                _volume_k[ _bdk[0, 0]:_bdk[1, 0], _bdk[0, 1]:_bdk[1, 1], _bdk[0, 2]:_bdk[1, 2]] = _gbk + _fk * subvol(_volume_k, _bdk)
                _volume_k = _volume_k.unsqueeze(0)
                if not(_foreground is None):
                    _volume_k = self.blendForeground(_volume_k, _foreground)                
                if _volumes is None:
                    _volumes = _volume_k
                else:
                    _volumes = torch.cat([_volumes, _volume_k])
            return _volumes            
    # Composer::blend     

    def blendGaussians(self, _G, _volume = None, _foreground = None):
        if _volume is None:
            _vol = self.zeros(self.shape)
        else:
            _vol = _volume

        if not torch.is_tensor(_G):
            _G = self.tensor(_G)  

        #G_render = self.drawGaussian(_G)
        #_vol = self.blend(G_render, _volume = _vol, blend_all = True)
        if self.volume_batch_size <= 1:
            for _Gi in _G:
                _Gi = _Gi.unsqueeze(0)
                G_render = self.drawGaussian(_Gi)    
                _vol = self.blend(G_render, _volume = _vol, blend_all = True)        
        else:
            batch_idx = vgi.batchIdx(_G.shape[0], self.volume_batch_size)
            for idx in batch_idx:
                #print('blendGaussians::idx', idx)
                _Gb = _G[idx[0]:idx[1]]
                G_render = self.drawGaussian(_Gb)                    
                _vol = self.blend(G_render, _volume = _vol, blend_all = True)

        if not(_foreground is None):
            _vol = self.blendForeground(_vol, _foreground)

        _tvol = self.data_trans(_vol)
        return _vol, _tvol

    # 
    def bestSearch(self, G, _volume, _foreground = None):
        G_render = self.drawGaussian(G)
        _G_vols = self.blend(G_render, _volume = _volume, _foreground = _foreground, blend_all = False)
        _G_tvols = None
        for _g_vol in _G_vols:
            _g_tvol = self.data_trans(_g_vol)
            _g_tvol = _g_tvol.unsqueeze(0)
            if _G_tvols is None:
                _G_tvols = _g_tvol
            else:
                _G_tvols = torch.cat([_G_tvols, _g_tvol])
        _G_losses = self.lf(_G_tvols)
        i_min = _G_losses.argmin()
        g = G[i_min]
        _vol = _G_vols[i_min]
        _tvol = _G_tvols[i_min]
        _loss = _G_losses[i_min]
        return g, _vol, _tvol, _loss, i_min
    # Composer::bestSearch

    def initGE(self, n_randoms, _volume,
        rand_batch_idx = None,
        loc_rate = 0.85,
        size_range = None,
        alpha_range = None,
        beta_range = None,
        beta_sampling = False, reference = None, sample_box_size = [3, 3, 3]):
        lf = self.lf
        if rand_batch_idx is None:
            rand_batch_idx = vgi.batchIdx(n_randoms, self.volume_batch_size)
        g_init_min, g_init_max = self.rangeInitG(size_range = size_range, alpha_range = alpha_range, beta_range = beta_range )
        G = np.random.uniform(g_init_min, g_init_max, (n_randoms, n_parameters))
        if beta_sampling:
            for g in G:
                g[-1] = self.betaSampling(g, reference, sample_box_size)

        g_best = None
        _best_vol = None
        _min_loss = None        
        for idx in rand_batch_idx:
            Gb = G[idx[0]:idx[1]]
            g_best_b, _vol_b, _tvol_b, _min_loss_b, i_min = self.bestSearch(Gb, self._volume)
            if _min_loss is None or _min_loss_b < _min_loss:
                g_best = g_best_b
                _min_loss = _min_loss_b
                _best_vol = _vol_b
        return g_best, _best_vol, _min_loss
    # Composer::initGE

    def initPolarGE(self, _volume, loccator, 
        z_range = None,
        radius_range = None,
        radian_range = None,
        size_range = None,
        alpha_range = None,
        beta_range = None,
        beta_sampling = False, reference = None, sample_box_size = [3, 3, 3]):

        g_init_min, g_init_max = self.rangeInitG(size_range = size_range, alpha_range = alpha_range, beta_range = beta_range )
        G = np.random.uniform(g_init_min, g_init_max, (n_randoms, n_parameters))
        if beta_sampling:
            for g in G:
                g[-1] = self.betaSampling(g, reference, sample_box_size)

        g_best = None
        _best_vol = None
        _min_loss = None        
        for idx in rand_batch_idx:
            Gb = G[idx[0]:idx[1]]
            g_best_b, _vol_b, _tvol_b, _min_loss_b, i_min = self.bestSearch(Gb, self._volume)
            if _min_loss is None or _min_loss_b < _min_loss:
                g_best = g_best_b
                _min_loss = _min_loss_b
                _best_vol = _vol_b
        return g_best, _best_vol, _min_loss
    # Composer::initGE    

    # ..............................................................
    # boxes: [slices, rows, columns, 2, 3]
    # idx:   [boxes, 3]
    # outputs:
    #   G: [n, 11]
    #   L: location boundary: [n, 2, 3], minimum and maximum points.
    #   S: the maximum size: [3]
    def initGridGE(self, boxes, idx, 
            size_f = 0.8, alpha = 0.8, 
            beta_sampling = False, reference = None, sample_box_size = [3, 3, 3]):
        n = idx.shape[0]
        G = np.zeros([n, 11])
        # alpha:
        G[:, 9] = alpha
        sel_boxes = None
        for idx_i in idx:
            z, y, x = idx_i
            box_i = boxes[None, z, y, x]
            if sel_boxes is None:
                sel_boxes = box_i
            else:
                sel_boxes = np.concatenate([sel_boxes, box_i])

        box_min = sel_boxes[:, 0] # [n, 3]
        box_max = sel_boxes[:, 1] # [n, 3]
        box_size = box_max - box_min # [n, 3]
        L = np.flip(sel_boxes, axis = -1) # [n, 3]; D, H, W => X, Y, Z
        S = np.flip(box_size, axis = -1)  # [n, 3]

        # location:
        G[:, 0:3] =  (L[:, 1] + L[:, 0]) / 2.0 
        # size:
        G[:, 3:6] = np.floor((S - 1.0) * size_f / self.area_f)  # e.g. floor((8 - 1) / 3) = 3
        # beta:
        if beta_sampling:
            for i in range(n):            
                G[i, -1] = self.betaSampling(G[i], reference, sample_box_size)
        G = np.clip(G, self.g_min, self.g_max)
        return G, L, S
    # Composer::initGridGE

    # ..................................................................
    # Assuming that no overlap exists between any two ellipsoids of G.
    # Although G is a batch of ellipsoids, this function sequentially optimizes the parameter vector of each ellipsoid.
    # G = [n, parameters]
    # _volume is the volume without G
    def hillClimb(self, G, _volume, _loss, 
        _g_min = None, _g_max = None,
        rounds = 100, acceleration = 1.2, min_decline = 0.00001, _foreground = None):
        n, n_parameters = G.shape 
        if torch.is_tensor(G):
            _G = G
        else:
            _G = self.tensor(G)      
        _step_size = vgi.clone(self._step_size).tile([n, 1]) #[n, parameters]
        _min_decline = self.tensor(min_decline)
        _acceleration = self.tensor(acceleration)
        _candidate = self.tensor([-acceleration, -1.0/acceleration, 1.0/acceleration, acceleration]) # [4]        
        _candidate = _candidate.unsqueeze(-1) # [4, 1]

        # the shapes of _g_min and _g_max can be [11] or [n, 11]
        if _g_min is None:
            _g_min = self._g_min
        if _g_max is None:
            _g_max = self._g_max
        lf = self.lf

        _G_cur = vgi.clone(_G)   
        _min_loss = _loss

        i_r = 0
        while i_r < rounds:
            _prev_loss = vgi.clone(_min_loss)
            for i_param in range(n_parameters):
                _G_cand = _G_cur.tile([4, 1, 1]) # [4, n, 11]
                #print('_G_cand', _G_cand.shape)
                _step = _candidate * _step_size[:, i_param] # [4, 1] * [n] = [4, n]
                #print('_step_size', _step_size.shape)
                #print('_step', _step.shape)
                #print('_G_cand[..., i_param]', _G_cand[..., i_param].shape)
                _G_cand[..., i_param] += _step    # [4, n, i] += [4, n] = [4, n, 11]
                _G_cand = _G_cand.clip(_g_min, _g_max) # clipping with [11] or [n, 11]
                _min_loss_u = vgi.clone(_min_loss)
                for i_g in range(n):
                    _g_u, _vol_u, _tvol_u, _loss_u, i_min = self.bestSearch(_G_cand[:, i_g], _volume, _foreground = _foreground)                    
                    #print('hillClimb', i_r, i_param, i_g, _loss_u, _min_loss_u)

                    if ( _loss_u < _min_loss_u ) :
                        _G_cur[i_g] = _g_u
                        _step_size[i_g, i_param] = _step[i_min, i_g] #bestStep 
                        _min_loss_u = _loss_u                
                    else:  # updating failed then shrinking            
                        _step_size[i_g, i_param] /= _acceleration 
                # g loop
            # Parameter loop      
            # Updating, all ellipsoids use the same _step_size
            #_volume, _tvol = self.blendGaussians(_G_cur, _volume, _foreground = None)
            #print('hillClimb::_tvol', _tvol.shape)            
            _min_loss = torch.min(_min_loss_u, _min_loss)          


            # Ending of each round            
            _loss_d = _prev_loss - _min_loss
            #print('hillClimb::_loss_d, i_r', _loss_d, i_r)
            if _loss_d >= self._zero and _loss_d < _min_decline:
                #print('hillClimb::end:_loss_d, i_r', _loss_d, i_r)
                i_r = rounds
            else:
                i_r += 1
        # Round loop
        _vol, _tvol = self.blendGaussians(_G_cur, _volume, _foreground = _foreground)
        return _G_cur, _vol, _min_loss
    # Composer::hillClimb
    # ..................................................................
    # Reconstruction 
    def reconstruct(self, n,
        n_randoms = 50, shrink_rate = 0.95, n_shrink = 10, 
        _volume = None, _foreground = None,
        init = 'random', # ['grid', 'random']
        loc_rate = 0.85,
        size_range = None,
        alpha_range = None,
        beta_range = None,        
        beta_sampling = False, reference = None, sample_box_size = [3, 3, 3],
        init_grid_thres = 0.001,
        min_decline = 0.0000001, opt_rounds = 100,                   
        verbose = 1, n_log = 100, log =  None): 

        # g = [[ux], [uy], [uz], [sx], [sy], [sz], [rx], [ry], [rz], [alpha], [beta]]
        # |g| = 11

        t_s = time.time()
        init_type = -1
        _g_min = self._g_min
        _g_max = self._g_max
        if init == 'grid':
            init_type = 0
            g_min_size = np.ceil(np.max(self.g_min[3:6]) * self.area_f * 2 + 1)
            min_vol_size = np.array([g_min_size, g_min_size, g_min_size])
            # divideBox will find the smallest power of 2 larger than g_min_size
            grid_B, grid_C, grid_idx = divideBox(self.box, min_size = min_vol_size)
                #   grid_B:   boxes:    [levels][slices, rows, columns, 2, 3]
                #   grid_C:   shapes:   [levles][slices, rows, columns]
                #   grid_idx: indeices: [levles][offests][boxes, 3]            
            init_grid_max_lv = len(grid_B) - 1
            init_grid_lv = 0
            init_grid_s = 0
            init_grid_sizef = 0.8 # the factor of size initial value.
            init_grid_alpha = (self.g_max[9] - self.g_min[9]) * 0.5
        elif init == 'random':
            init_type = 1

        lf = self.lf
        opt = self.hillClimb
        rand_batch_idx = vgi.batchIdx(n_randoms, self.volume_batch_size)
        sample_box_size = np.array(sample_box_size)
        _G = None
        if _volume is None:
            _volume = self._volume   
        else:
            if not torch.is_tensor(_volume):
                _volume = self.tensor(_volume)      
        n_g = 0
        n_shrink_g = 0
        n_log_g = 0
        n_G_batch = 0
        _loss_prev = lf(self.data_trans(_volume).unsqueeze(0))
        _loss = None
        while n_g < n:   
            update_flag = True    
            if init_type == 0:

                # G_batch: [n, 11]
                # loc_bds: [n, 2, 3], location boundaries (x, y, z)
                # size_bds: [n, 3], size boundaries (x, y, z)

                G_batch, loc_bds, size_bds = self.initGridGE(boxes = grid_B[init_grid_lv], 
                    idx = grid_idx[init_grid_lv][init_grid_s],
                    size_f = init_grid_sizef, alpha = init_grid_alpha, 
                    beta_sampling = beta_sampling, reference = reference, sample_box_size = sample_box_size)
                n_G_batch = G_batch.shape[0]

                _vol_k, _tvol_k = self.blendGaussians(G_batch, _volume)
                _loss_k = lf(_tvol_k)

                #print('reconstruct::init_grid_lv, init_grid_s, n_G_batch:', init_grid_lv, init_grid_s, n_G_batch)
                # optimizing boundaries
                _g_min = self._g_min.tile([n_G_batch, 1])
                _g_max = self._g_max.tile([n_G_batch, 1])
                # location:
                _g_min[:, 0:3] = self.tensor(np.array(loc_bds[:, 0]))
                _g_max[:, 0:3] = self.tensor(np.array(loc_bds[:, 1]))
                # size 
                _g_max[:, 3:6] = self.tensor(np.array(size_bds))

                init_grid_s += 1
                if init_grid_s >= len(grid_idx[init_grid_lv]):
                    init_grid_s = 0
                    if not(_loss is None):
                        _loss_d = (_loss_prev - _loss).abs()
                        #print('reconstruct::_loss_d', _loss_d)
                        # Go to the next level
                        if _loss_d < init_grid_thres:                            
                            init_grid_lv = init_grid_max_lv if init_grid_lv >= init_grid_max_lv else init_grid_lv + 1
                            #print('reconstruct::init_grid_lv', init_grid_lv)
            elif init_type == 1:
                g_k, _vol_k, _loss_k = self.initGE(n_randoms = n_randoms, _volume = _volume, 
                    rand_batch_idx = rand_batch_idx,
                    loc_rate = loc_rate, size_range = size_range, 
                    alpha_range = alpha_range, beta_range = beta_range,
                    beta_sampling = beta_sampling, reference = reference, sample_box_size = sample_box_size)
                G_batch = np.expand_dims(g_k, 0) # [parameters] -> [1, parameters] 
            # init_type?

            if not(_loss is None):
                _loss_prev = vgi.clone(_loss)            

            n_G_batch = G_batch.shape[0]
            if update_flag or n_G_batch >= self.max_batch_size:
                # Optimizing and updating
                _G_batch_opt, _volume, _loss = opt(G_batch, _volume, _loss_k, 
                    _g_min = _g_min, _g_max = _g_max,
                    _foreground = _foreground, min_decline = min_decline, rounds = opt_rounds)
                #print('reconstruct::_loss, _loss_prev:', _loss, _loss_prev)

                #_volume = _vol_k # For debug, remove this line!
                self.setVolume(_volume)
                if _G is None:
                    _G = _G_batch_opt
                else:
                    _G = torch.cat([_G, _G_batch_opt])

                # Post processing                
                n_g += n_G_batch
                n_log_g += n_G_batch
                n_shrink_g += n_G_batch
                if n_shrink_g >= n_shrink:
                    if not (size_range is None):
                        size_range *= shrink_rate
                    n_shrink_g = 0
                if n_log_g >= n_log:
                    t = time.time() - t_s
                    print('#Ellipsoids:', n_g, ', loss:', _loss, ', time:', t, '(', vgi.currentTime(), ')', flush=True)
                    #print('size_range', size_range)
                    n_log_g = 0
                # Clearing
                G_batch = None
                n_G_batch = 0
            # if n_G_batch >= self.max_batch_size:
        # while k < n
        torch.cuda.empty_cache()
        return _G, _volume, _loss
    # Composer::reconstruct     

    # Reoptimization
    def reoptimize(self, _G, _volume, _tvol = None, _loss = None,
        min_decline = 0.0000001, opt_rounds = 100, clamp = True,                  
        verbose = 1, n_log = 100, log =  None):
        t_s = time.time()
        n = _G.shape[0]
        lf = self.lf
        opt = self.hillClimb

        if _tvol is None:
            _tvol = self.data_trans(_volume)
        if _loss is None:
            _loss = lf(_tvol)

        _vol_bg = vgi.clone(_volume)
        _vol_fg = self.zeros(_volume.shape)
        _f_kp = self.ones(_volume.shape)       

        _G_out = None
        k = n - 1
        n_g = 0
        n_log_g = 0
        batch_size = self.volume_batch_size
        ib_s = n - batch_size
        ib_e = n         
        while k >= 0:
            ib_s = max(ib_s, 0) 
            n_b = ib_e - ib_s
            #print('reoptimize::ib_s:ib_e', ib_s, ib_e)
            _Gb = _G[ib_s:ib_e]
            #_I_Ga, _f_each, _G = self.drawGaussian(_gb)
            Gb_render = self.drawGaussian(_Gb)
            kb = n_b - 1
            while kb >= 0:                
                #_I_Ga_k = _I_Ga[kb].unsqueeze(0)
                #_f_k = _f_each[kb].unsqueeze(0)
                _gbk, _fk, _gwk, _bdk = Gb_render[kb]
                

                #print('reoptimize::_vol_bg', _vol_bg.shape)
                #print('reoptimize::_gb_k', _gb_k.shape)
                #print('reoptimize::_f_k', _f_k.shape)
                #_vol_bg = (_vol_bg - _gb_k) / _f_k

                _vol_bg[ _bdk[0, 0]:_bdk[1, 0], _bdk[0, 1]:_bdk[1, 1], _bdk[0, 2]:_bdk[1, 2]] = (subvol(_vol_bg, _bdk) - _gbk) / _fk

                if clamp:
                    torch.clamp(_vol_bg, min=0.0, max=1.0, out=_vol_bg)

                _foreground = (_vol_fg, _f_kp)

                # Updating  
                _Gk = _Gb[kb].unsqueeze(0)
                #print('reoptimize::k', k, _Gk, _bdk)
                _Gku, _volume, _loss = opt(_Gk, _vol_bg, _loss, _foreground = _foreground, min_decline = min_decline, rounds = opt_rounds)

                if _G_out is None:
                    _G_out = _Gku
                else:
                    _G_out = torch.cat([_Gku, _G_out])

                Gku_render = self.drawGaussian(_Gku)
                _gbku, _fku, _gwku, _bdku = Gku_render[0]
                #print('reoptimize::ku', k, _Gku, _bdku)
                
                # Forward alpha compositing: J_k = J_{k - 1} + \beta_k \alpha_k G_k * f_{k - 1}
                _vol_fg[ _bdku[0, 0]:_bdku[1, 0], _bdku[0, 1]:_bdku[1, 1], _bdku[0, 2]:_bdku[1, 2]] += _gbku * subvol(_f_kp, _bdku)
                _f_kp[ _bdku[0, 0]:_bdku[1, 0], _bdku[0, 1]:_bdku[1, 1], _bdku[0, 2]:_bdku[1, 2]] *= _fku

                n_g += 1
                n_log_g += 1
                kb -= 1
                k -= 1 
                if n_log_g >= n_log:
                    t = time.time() - t_s
                    print('BO G:', n_g, ', loss:', _loss, ', time:%0.2f'%t, '(', vgi.currentTime(), ')', flush=True)
                    n_log_g = 0                  
                if n_g >= n:
                    break; 
            # batch loop
            ib_e -= n_b 
            ib_s -= batch_size 
            if n_g >= n:
                break;         
        # g loop

        if not(log is None):
            log += [log_t]

        torch.cuda.empty_cache()
        return _G_out, _volume, _loss
    # Composer::reoptimize    

    # ..................................................................
    # Reconstruction with the initialization in polar space
    def reconstructPolar(self, n, 
        n_radians, n_radii, n_slabs, n_rounds,
        n_randoms = 5, 
        optimize = True, 
        sampling = None,
        shrink_rate = 0.95, n_shrink = 10, 
        _volume = None, _foreground = None,
        size_range = None,
        alpha_range = None,
        beta_range = None,        
        beta_sampling = False, reference = None, sample_box_size = [3, 3, 3],
        init_grid_thres = 0.001,
        min_decline = 0.0000001, opt_rounds = 100,                   
        verbose = 1, n_log = 100, log =  None): 

        sample_box_size = np.array(sample_box_size)
        
        if _volume is None:
            _volume = self._volume   
        else:
            if not torch.is_tensor(_volume):
                _volume = self.tensor(_volume)

        # render
        def render(g):
            _G = self.tensor(g).unsqueeze(0)
            G_render = self.drawGaussian(_G)
            _volume_g = self.blend(G_render, _volume = _volume, _foreground = _foreground, blend_all = True)                        
            _loss_g = lf(self.data_trans(_volume_g).unsqueeze(0))  
            return  _volume_g, _loss_g

        def transg(g):
            # 2D polar -> Cartesian
            radian, radius = g[0:2]
            x = radius * np.cos(radian)
            y = radius * np.sin(radian)
            g[0], g[1] = x, y 
            # Beta
            g[-1] = self.betaSampling(g, reference, sample_box_size)   
            return g                 

        # Smaplers
        def meanSampler(ga, gb):
            g_j = (ga + gb) / 2
            g_j = transg(g_j)
            _volume_gj, _loss_gj = render(g)
            return g_j, _volume_gj, _loss_gj

        def randomSampler(ga, gb):
            _loss_min_smp = 9999999999.99
            g_out = None
            _volume_out = None
            for i in range(n_randoms):
                g_j = np.random.uniform(ga, gb)
                g_j = transg(g_j)                
                _volume_gj, _loss_gj = render(g_j) 
                if _loss_gj < _loss_min_smp:
                    _loss_min_smp = _loss_gj
                    _volume_out = _volume_gj
                    g_out = g_j
            return g_out, _volume_out, _loss_min_smp

        if sampling is None:
            sampler = meanSampler
        elif sampling == 'random':
            sampler = randomSampler

        t_s = time.time()
        _g_min = self._g_min
        _g_max = self._g_max

        lf = self.lf
        opt = self.hillClimb
        _G = None
        n_g = 0
        n_shrink_g = 0
        n_log_g = 0
        n_G_batch = 0
        _loss_prev = None
        _loss = None
        
        zd = self.shape[0] / n_slabs
        radius_1 = max(self.shape[1], self.shape[2]) / (n_radii ** 0.5)
        td = np.pi * 2 / n_radians

        # g = [t, r, z, sx, sy, sz, rx, ry, rz, alpha, beta]
        # |g| = 11
        g_init_min = np.array(self.g_min)
        g_init_max = np.array(self.g_max)

        # Size
        g_init_min[3:6] = size_range[0]
        g_init_max[3:6] = size_range[1]
        # alpha
        g_init_min[-2] = alpha_range[0]
        g_init_max[-2] = alpha_range[1]      
        for i_round in range(n_rounds):
            i_radius = n_radii
            while i_radius > 0:
            #i_radius = 1
            #while i_radius <= n_radii :
                # Radius range
                g_init_min[1] = radius_1 * (i_radius - 1) ** 0.5 
                g_init_max[1] = radius_1 * (i_radius) ** 0.5 
                # Slab                 
                g_init_min[2], g_init_max[2] = self.min_p[0], self.min_p[0] + zd 
                for i_slab in range(n_slabs):   
                    G_batch = None             
                    # Radian
                    g_init_min[0], g_init_max[0] = 0.0, td  
                    for i_radian in range(n_radians):
                        gk, _volume_gk, _loss_gk = sampler(g_init_min, g_init_max) # [11, ]
                        #if _loss_prev is None or _loss_gk <= _loss_prev: 
                        if True:     
                            gk = np.expand_dims(gk, 0) # [1, 11]
                            if G_batch is None:
                                G_batch = gk
                            else:
                                G_batch = np.concatenate([G_batch, gk])
                            g_init_min[0], g_init_max[0] = g_init_max[0], g_init_max[0] + td 
                    # Radian loop
                    if G_batch is None:
                        n_G_batch = 0
                    else:
                        n_G_batch = G_batch.shape[0]
                    if n_G_batch > 0:
                        _G_batch = self.tensor(G_batch) # [n_G_batch, parameters]
                        if not optimize:
                            # Drawing 
                            G_render = self.drawGaussian(_G_batch)
                            _volume = self.blend(G_render, _volume = _volume, _foreground = _foreground, blend_all = True)                        
                            _loss = lf(self.data_trans(_volume).unsqueeze(0)) 

                            if _G is None:
                                _G = _G_batch
                            else:
                                _G = torch.cat([_G, _G_batch])

                        else: 
                            for _gk in _G_batch: 
                                _gk = _gk.unsqueeze(0)
                                G_render = self.drawGaussian(_gk)
                                _volume_k = self.blend(G_render, _volume = _volume, _foreground = _foreground, blend_all = True)                        
                                _loss_k = lf(self.data_trans(_volume_k).unsqueeze(0))     

                                if _loss_prev is None:
                                    _loss_prev = _loss_k
                                elif not(_loss is None):
                                    _loss_prev = _loss
         
                                # Optimizing and updating
                                _gk_opt, _volume, _loss = opt(_gk, _volume, _loss_k, 
                                    _g_min = _g_min, _g_max = _g_max,
                                    _foreground = _foreground, min_decline = min_decline, rounds = opt_rounds)
                                #print('reconstructPolar::_loss, _loss_k, _loss_prev:', _loss, _loss_k, _loss_prev)
                                self.setVolume(_volume)
                                if _G is None:
                                    _G = _gk_opt
                                else:
                                    _G = torch.cat([_G, _gk_opt])
                            # batch loop
                        # optimize? 
                        n_g += n_G_batch
                        n_log_g += n_G_batch
                        n_shrink_g += n_G_batch
                        if n_shrink_g >= n_shrink:
                            if not (size_range is None):
                                size_range *= shrink_rate
                                g_init_min[3:6] = size_range[0]
                                g_init_max[3:6] = size_range[1] 
                                #print('i_radius, i_slab:', i_radius, i_slab)   
                                print('shrinked:', size_range)                        
                            n_shrink_g = 0
                        if n_log_g >= n_log:
                            t = time.time() - t_s
                            print('#Ellipsoids:', n_g, ', loss:', _loss, ', time:', t, '(', vgi.currentTime(), ')', flush=True)
                            #print('size_range', size_range)
                            n_log_g = 0

                    # n_G_batch > 0?                
                    g_init_min[2], g_init_max[2] = g_init_max[2], g_init_max[2] + zd
                    if n_g >= n: # Stop!
                        break                         
                # slab loop  
                i_radius -= 1
                #i_radius += 1
                if n_g >= n: # Stop!
                    break                 
            # radius loop   
            if n_g >= n: # Stop!
                break                   
        # round loop
        torch.cuda.empty_cache()
        return _G, _volume, _loss
    # Composer::reconstructPolar       


    # reconstructErrMap
    # vol is a numpy.ndarray representing a pre-reconstrcuted volume.  
    def reconstructErrMap(self, vol, gt = None,
                           clip_min = 0.0, clip_max = None, sigma = 1.0,
                           bin_threshold = 0.4, bin_threshold_ratio = 0.9, 
                           min_vx = 7, min_init_size = 5.0, 
                           rep = 2, rep_vx = 27, rep_s = 0.5,
                           init_opt = True, # Initial optimization for each g
                           n_randoms = 5,   # init_opt must be True if n_randoms > 0 
                           epoches = 3,     # The #iterations for G batch generating
                           rounds = 10,         # The #iterations for optimizing each G batch
                           opt_rounds = 100,        # The #iterations for each optimization
                            min_decline = 0.0000001,  clamp = True,                  
                            verbose = 1, n_log = 100, log =  None):
        if verbose & 1:
            if not(gt is None):
                ev = evaluate(vol, gt, adjust = True)  
                print('Init ev:', strev(ev), flush = True)    
        if verbose & 2:
            if not(gt is None): 
                print('ground truth:')
                show(gt)
            print('vol:')
            show(vol)
        _vol = self.tensor(vol)       
        target_proj = np.swapaxes(self.target, 0, 1) # (angle, H, W) => (H, angle, W)
        self._G = None
        for i_e in range(epoches):
            print('Epoch', i_e)
            ts_init = time.time()
            G_e = errorGaussians(vol, self.data_min_p, target_proj, self.data_trans, 
                           min_init_size = min_init_size,
                           clip_min = clip_min, clip_max = clip_max, sigma = sigma, 
                           bin_threshold = bin_threshold, min_vx = min_vx,
                           rep = rep, rep_vx = rep_vx, rep_s = rep_s)
            
            if G_e is None or len(G_e) == 0:
                print('G_e is None or len(G_e) == 0')
                break
            if verbose & 1:    
                t = time.time() - ts_init            
                print('Init G_e:', G_e.shape, ', time:%0.2f'%t)
            n_g = len(G_e)
            _G_e = self.tensor(G_e)
            if init_opt: # .........
                lf = self.lf
                opt = self.hillClimb
                _g_min = self._g_min
                _g_max = self._g_max  
                _G_e_opt = None       
                n_log_g = 0    
                t_s = time.time() 
                for i_g_k, _g_k in enumerate(_G_e):
                    if n_randoms > 0:
                        _loss_k = 9999999999.99
                        g_k0 = G_e[i_g_k]
                        _g_k = None
                        _vol_k = None
                        _tvol_k = None
                        for j in range(n_randoms):
                            g_init_min = np.array(self.g_min)
                            g_init_max = np.array(self.g_max)    
                            # beta
                            g_init_min[-1] = np.clip(g_k0[-1] - 0.1, self.g_min[-1], None)
                            # alpha
                            g_init_max[-2] = np.clip(g_k0[-2] + 0.1, None, self.g_max[-2])
                            g_j = np.random.uniform(g_init_min, g_init_max)

                            g_j[:6] = g_k0[:6]
 

                            _g_j = self.tensor(g_j).unsqueeze(0)
                            _vol_j, _tvol_j = self.blendGaussians(_g_j, _vol)
                            _loss_j = lf(_tvol_j) 

                            if _loss_j < _loss_k:
                                _loss_k = _loss_j
                                _vol_k = _vol_j
                                _tvol_k = _tvol_j
                                _g_k = _g_j                 
                    else:
                        _g_k = _g_k.unsqueeze(0)
                        _vol_k, _tvol_k = self.blendGaussians(_g_k, _vol)
                        _loss_k = lf(_tvol_k)


                    # Optimizing and updating
                    _g_k_opt, _vol, _loss = opt(_g_k, _vol, _loss_k, 
                        _g_min = _g_min, _g_max = _g_max,
                        _foreground = None, min_decline = min_decline, rounds = opt_rounds)
                    if _G_e_opt is None:
                        _G_e_opt = _g_k_opt
                    else:
                        _G_e_opt = torch.cat([_G_e_opt, _g_k_opt])
                    if verbose & 1:
                        n_log_g += 1
                        if n_log_g >= n_log:
                            t = time.time() - t_s
                            print('FO G:', _G_e_opt.shape[0], ', loss:', _loss, ', time:%0.2f'%t, '(', vgi.currentTime(), ')', flush=True)
                            n_log_g = 0                            
                # loop _G_e

                _loss_r0 = _loss * 2
                _vol_r = _vol
                _G_r = _G_e_opt
            # init_opt .............
            else:
                _vol, _tvol = self.blendGaussians(_G_e, _vol)
                _loss_r0 = self.lf(_tvol)
                _vol_r = _vol
                _G_r = _G_e
            # No init opt 

            if verbose & 1:
                t = time.time() - t_s
                print('FO finished', ', #g:', _G_r.shape[0], ', time: %0.2f'%t, '(', vgi.currentTime(), ')', flush=True)                
                #print('Epoch', i_e, 'loss:', _loss_r0, ', #g:', _G_r.shape[0], flush = True)
                if not(gt is None):
                    vol_r = vgi.toNumpy(_vol_r)
                    ev = evaluate(vol_r, gt, adjust = True)  
                    print('FO ev:', strev(ev), flush = True)                  
                    if verbose & 2:
                        show(vol_r)

            for i_r in range(rounds):
                ts_bo = time.time()
                _G_r, _vol_r, _loss_r = self.reoptimize(_G_r, _vol_r, _loss = _loss_r0,
                                        min_decline = min_decline, opt_rounds = opt_rounds, 
                                        clamp = clamp, verbose = verbose, n_log = n_log, log = log)
                if verbose & 1:
                    if not(gt is None):
                        vol_r = vgi.toNumpy(_vol_r)
                        t = time.time() - ts_bo
                        ev = evaluate(vol_r, gt, adjust = True)  
                        print('BO finished [', i_e, ':', i_r, ']', ', time:%0.2f'%t, ', ev:', strev(ev), flush = True)                  
                        if verbose & 2:
                            show(vol_r)
                _opt_d = _loss_r0 - _loss_r
                if _opt_d < min_decline:
                    print('Early stop:', _opt_d)
                    break
            # rounds
            _vol = _vol_r
            vol = vgi.toNumpy(_vol)
            bin_threshold *= bin_threshold_ratio
            if self._G is None:
                self._G = _G_r
            else:
                self._G = torch.cat([self._G, _G_r])
        # epoches   
        return vol, self._G

    # Composer::reconstructErrMap