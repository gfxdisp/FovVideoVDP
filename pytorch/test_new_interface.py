from fvvdp import *

w, h = 512, 256
frames = 10

test_img = torch.ones( [frames,3,w,h], dtype=torch.float32 ) * 0.5
ref_img = test_img.clone() + (torch.rand( test_img.shape, dtype=torch.float32 )-0.5)*0.2

vs = fvvdp_video_source_dm( test_img, ref_img, 30, color_space_name='sRGB' )
