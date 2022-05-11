from fvvdp import *

from gfxdisp.pfs.pfs_torch import pfs_torch

w, h = 512, 256
frames = 10

test_img = torch.ones( [frames,3,w,h], dtype=torch.float32 ) * 0.5
ref_img = test_img.clone() + (torch.rand( test_img.shape, dtype=torch.float32 )-0.5)*0.2

#vs = fvvdp_video_source_dm( test_img, ref_img, 30, dims="FCWH", color_space_name='sRGB' )

# pfs_torch.view( vs.get_test_frame(0) )
# pfs_torch.view( vs.get_reference_frame(0) )

fvv = fvvdp(display_name="standard_4k", display_photometry=None, display_geometry=None, color_space="sRGB", foveated=False, heatmap=None, quiet=False)

(Q_JOD, diff_map) = fvv.predict( test_img, ref_img, dim_order="FCWH", frames_per_second=30 )

print( "Q_JOD: {}".format(Q_JOD) )



