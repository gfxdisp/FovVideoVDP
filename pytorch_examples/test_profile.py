# Profile PyTorch execution

import os
import glob
import time

from torch.profiler import profile, record_function, ProfilerActivity

import pyfvvdp

display_name = 'standard_hdr'
media_folder = os.path.join(os.path.dirname(__file__), '..',
                            'example_media', 'aliasing')

# Add here paths to the test and reference videos
ref_fname = 'S:\\Datasets\\LIVEHDR\\test\\4k_ref_Bonfire.mp4'
tst_fname = 'S:\\Datasets\\LIVEHDR\\test\\4k_6M_Bonfire.mp4'

fv = pyfvvdp.fvvdp(display_name=display_name, heatmap=None)

vs = pyfvvdp.fvvdp_video_source_file( tst_fname, ref_fname, display_photometry=display_name, frames=10 )

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        start = time.time()
        Q_JOD_static, stats_static = fv.predict_video_source( vs )
        end = time.time()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
#print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
prof.export_chrome_trace("trace.json")

print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
