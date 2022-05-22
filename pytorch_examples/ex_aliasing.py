# This example shows how to use python interface to run FovVideoVDP directly on video files
import glob
import time
from pyfvvdp import fvvdp
from pyfvvdp.video_source_file import fvvdp_video_source_file

display_name = 'sdr_fhd_24';

ref_file = 'pytorch_examples/aliasing/ferris-ref.mp4';
TST_FILEs = glob.glob( 'pytorch_examples/aliasing/ferris-*-*.mp4' );

fv = fvvdp(display_name=display_name, heatmap=None)

for tst_fname in TST_FILEs:

    vs = fvvdp_video_source_file( tst_fname, ref_file, display_photometry=display_name )

    start = time.time()
    Q_JOD_static, stats_static = fv.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
