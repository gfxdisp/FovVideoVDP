
import os
from third_party.loadmat import loadmat
import torch
import numpy as np
import json
import torch.nn.functional as Func
from PIL import Image

def torch_gpu_mem_info():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    print("GPU mem used: %d M (cache %d M)" % (a/(1024*1024), c/(1024*1024)))

def json2dict(file):
    data = None
    if os.path.isfile(file):
        with open(file, "r") as json_file:
            data=json.load(json_file)
    else:
        print("Error: Cannot find file %s" % file)
    return data

def linear2srgb_torch(lin):
    lin = torch.clamp(lin, 0.0, 1.0)
    srgb = torch.where(lin > 0.0031308, (1.055 * (lin ** (1/2.4))) - 0.055, 12.92 * lin)
    return srgb

def srgb2linear_torch(srgb):
    srgb = torch.clamp(srgb, 0.0, 1.0)
    lin = torch.where(srgb > 0.04045, ((srgb + 0.055) / 1.055)**2.4, srgb/12.92)
    return lin

def img2np(img):
    return np.array(img, dtype="float32") * 1.0/255.0

def np2img(nparr):
    return Image.fromarray(np.clip(nparr * 255.0, 0.0, 255.0).astype('uint8'))

def l2rgb(x):
    return np.concatenate([x,x,x], -1)

def stack_horizontal(nparr):
    return np.concatenate([nparr[i] for i in range(len(nparr))], axis=-2)

def stack_vertical(nparr):
    return np.concatenate([nparr[i] for i in range(len(nparr))], axis=-3)


def load_mat_dict(filepath, data_label, device):
    # datapath = "D:\\work\\st_fov_metric"
    # filepath = os.path.join(datapath, rel_path)

    if not os.path.isfile(filepath):
        return None
    else:
        v = loadmat(filepath)
        if data_label in v:
            return v[data_label]
        else:
            print("Cannot find key %s, valid keys are %s" % (data_label, v.keys()))
            return None

def load_mat_tensor(filepath, data_label, device):
    # datapath = "D:\\work\\st_fov_metric"
    # filepath = os.path.join(datapath, rel_path)

    if not os.path.isfile(filepath):
        return None
    else:
        v = loadmat(filepath)
        if data_label in v:
            return torch.tensor(v[data_label], device=device)
        else:
            print("Cannot find key %s, valid keys are %s" % (data_label, v.keys()))
            return None


# args are indexed at 0
def fovdots_load_ref(content_id, device, data_res="full"):
    if data_res != "full":
        print("Note: Using data resolution %s" % (data_res))
    hwd_tensor = load_mat_tensor("D:\\work\\st_fov_metric\\data_vid_%s\\content_%d_ref.mat" % (data_res, content_id+1), "I_vid", device)
    dhw_tensor = hwd_tensor.permute(2,0,1)
    ncdhw_tensor = torch.unsqueeze(torch.unsqueeze(dhw_tensor, 0), 0)
    return ncdhw_tensor

# args are indexed at 0
def fovdots_load_condition(content_id, condition_id, device, data_res="full"):
    if data_res != "full":
        print("Note: Using data resolution %s" % (data_res))
    hwd_tensor = load_mat_tensor("D:\\work\\st_fov_metric\\data_vid_%s\\content_%d_condition_%d.mat" % (data_res, content_id+1, condition_id+1), "I_vid", device)
    dhw_tensor = hwd_tensor.permute(2,0,1)
    ncdhw_tensor = torch.unsqueeze(torch.unsqueeze(dhw_tensor, 0), 0)
    return ncdhw_tensor


class ImGaussFilt():
    def __init__(self, sigma, device):
        self.filter_size = 2 * int(np.ceil(2.0 * sigma)) + 1
        self.half_filter_size = (self.filter_size - 1)//2

        self.K = torch.zeros((1, 1, self.filter_size, self.filter_size), device=device)

        for ii in range(self.filter_size):
            for jj in range(self.filter_size):
                distsqr = float(ii - self.half_filter_size) ** 2 + float(jj - self.half_filter_size) ** 2
                self.K[0,0,jj,ii] = np.exp(-distsqr / (2.0 * sigma * sigma))

        self.K = self.K/self.K.sum()

    def run(self, img):
        
        if len(img.shape) == 2: img_4d = img.reshape((1,1,img.shape[0],img.shape[1]))
        else:                   img_4d = img

        pad = (
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,)

        img_4d = Func.pad(img_4d, pad, mode='reflect')
        return Func.conv2d(img_4d, self.K)[0,0]