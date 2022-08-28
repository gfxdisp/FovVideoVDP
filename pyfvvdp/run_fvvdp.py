# Command-line interface for FovVideoVDP. 

import os, sys
import os.path
import argparse
import logging
#from natsort import natsorted
import glob
import ffmpeg
import numpy as np
import torch
import imageio.v2 as imageio
#from PIL import Image
from pyfvvdp.video_source_file import fvvdp_video_source_file
from pyfvvdp.fvvdp import fvvdp
from pyfvvdp.visualize_diff_map import visualize_diff_map
#from pytorch_msssim import SSIM
from utils import *

def expand_wildcards(filestrs):
    if not isinstance(filestrs, list):
        return [ filestrs ]
    files = []
    for filestr in filestrs:
        if "*" in filestr:
            curlist = glob.glob(filestr)
            files = files + curlist
        else:
            files.append(filestr)
    return files

# Save a numpy array as a video
def np2vid(np_srgb, vidfile, fps, verbose=False):

    N, H, W, C = np_srgb.shape
    if C == 1:
        np_srgb = np.concatenate([np_srgb]*3, -1)
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(W, H), r=fps)
            .output(vidfile, pix_fmt='yuv420p', crf=10)
            .overwrite_output()
            .global_args( '-hide_banner')
            .global_args( '-loglevel', 'info' if verbose else 'quiet')
            .run_async(pipe_stdin=True)
    )
    for fid in range(N):
        process.stdin.write(
                (np_srgb[fid,...] * 255.0)
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

# Save a numpy array as an image
def np2img(np_srgb, imgfile):

    N, H, W, C = np_srgb.shape
    if C == 1:
        np_srgb = np.concatenate([np_srgb]*3, -1)

    if N>1:
        sys.exit("Expecting an image, found video")

    imageio.imwrite( imgfile, (np.clip(np_srgb,0.0,1.0)[0,...]*255.0).astype(np.uint8) )

# -----------------------------------
# Command-line Arguments
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FovVideoVDP on a set of videos")
    parser.add_argument("--test", type=str, nargs='+', required = True, help="list of test images/videos")
    parser.add_argument("--ref", type=str, nargs='+', required = True, help="list of reference images/videos")
    parser.add_argument("--gpu", type=int,  default=-1, help="select which GPU to use (e.g. 0), default is CPU")
    parser.add_argument("--heatmap", type=str, default="none", help="type of difference map (none, raw, threshold, supra-threshold)")
    parser.add_argument("--heatmap-dir", type=str, default=None, help="in which directory heatmaps should be stored (the default is the current directory)")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose mode")
    parser.add_argument("--foveated", action='store_true', default=False, help="Run in a foveated mode (non-foveated is the default)")
    parser.add_argument("--display", type=str, default="standard_4k", help="display name, e.g. HTC Vive")
    #parser.add_argument("--nframes", type=int, default=60, help="# of frames from video you want to load")
    parser.add_argument("--quiet", action='store_const', const=True, default=False, help="Do not print any information but the final JOD value.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
        
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=log_level)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    logging.info("Running on device: " + str(device))

    heatmap_types = {
        "threshold"   : {"scale" : 1.000, "colormap_type": "trichromatic"},
        "supra-threshold" : {"scale" : 0.333, "colormap_type": "dichromatic"},
    }

    if args.heatmap == "none":
        args.heatmap = None

    if args.heatmap:
        if not args.heatmap in heatmap_types:
            logging.error( 'The recognized heatmap types are: "none", "threshold" and "supra-threshold"' )
            sys.exit()

        do_heatmap = True
    else:
        do_heatmap = False

    args.test = expand_wildcards(args.test)
    args.ref = expand_wildcards(args.ref)    

    N_test = len(args.test)
    N_ref = len(args.ref)

    if N_test==0:
        logging.error( "No test images/videos found." )
        sys.exit()

    if N_ref==0:
        logging.error( "No reference images/videos found." )
        sys.exit()

    if N_test != N_ref and N_test != 1 and N_ref != 1:
        logging.error( "Pass the same number of reference and test sources, or a single reference (to be used with all test sources), or a single test (to be used with all reference sources)." )
        sys.exit()

    fv = fvvdp( display_name=args.display, foveated=args.foveated, heatmap=args.heatmap, device=device )

    logging.info( 'When reporting metric results, please include the following information:' )    

    if args.display.startswith('standard_'):
        #append this if are using one of the standard displays
        standard_str = ', (' + args.display + ')'
    else:
        standard_str = ''
    fv_mode = 'foveated' if args.foveated else 'non-foveated'
    logging.info( '"FovVideoVDP v{}, {:.4g} [pix/deg], Lpeak={:.5g}, Lblack={:.4g} [cd/m^2], {}{}"'.format(fv.version, fv.pix_per_deg, fv.display_photometry.get_peak_luminance(), fv.display_photometry.get_black_level(), fv_mode, standard_str) )

    for kk in range( max(N_test, N_ref) ):
        test_file = args.test[min(kk,N_test-1)]
        ref_file = args.ref[min(kk,N_ref-1)]
        logging.info("Predicting the quality of '" + test_file + "' compared to '" + ref_file + "' ...")
        vs = fvvdp_video_source_file( test_file, ref_file )
        Q_jod, stats = fv.predict_video_source(vs)
        if args.quiet:                
            print( "{Q_jod:0.4f}".format(Q_jod=Q_jod) )
        else:
            print( "Q_JOD={Q_jod:0.4f}".format(Q_jod=Q_jod) )

        if do_heatmap:
            diff_type = heatmap_types[args.heatmap]
            # heatmap = stats["heatmap"] * diff_type["scale"]
            # diff_map_viz = visualize_diff_map(heatmap, context_image=ref_vid_luminance, colormap_type=diff_type["colormap_type"])
            out_dir = "." if args.heatmap_dir is None else args.heatmap_dir
            os.makedirs(out_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(test_file))            
            if stats["heatmap"].shape[2]>1: # if it is a video
                dest_name = os.path.join(out_dir, base + "_heatmap.mp4")
                logging.info("Writing heat map '" + dest_name + "' ...")
                np2vid(torch.squeeze(stats["heatmap"].permute((2,3,4,1,0)), dim=4).cpu().numpy(), dest_name, vs.get_frames_per_second(), args.verbose)
            else:
                dest_name = os.path.join(out_dir, base + "_heatmap.png")
                logging.info("Writing heat map '" + dest_name + "' ...")
                np2img(torch.squeeze(stats["heatmap"].permute((2,3,4,1,0)), dim=4).cpu().numpy(), dest_name)
                
            del stats

    #     del test_vid
    #     torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
