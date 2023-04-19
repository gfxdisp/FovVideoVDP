# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
import time

import pyfvvdp

display_name = 'standard_fhd'
media_folder = os.path.join(os.path.dirname(__file__), '..',
                            'example_media', 'aliasing')

ref_file = os.path.join(media_folder, 'ferris-ref.mp4')
TST_FILEs = glob.glob(os.path.join(media_folder, 'ferris-*-*.mp4'))

fv = pyfvvdp.fvvdp(display_name=display_name, heatmap=None)

for tst_fname in TST_FILEs:

    vs = pyfvvdp.fvvdp_video_source_file( tst_fname, ref_file, display_photometry=display_name )

    start = time.time()
    Q_JOD_static, stats_static = fv.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
