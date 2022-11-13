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

import pyfvvdp

from pyfvvdp.fvvdp_display_model import fvvdp_display_photometry, fvvdp_display_geometry
# from pyfvvdp.visualize_diff_map import visualize_diff_map
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
    parser.add_argument("--test", type=str, nargs='+', required = False, help="list of test images/videos")
    parser.add_argument("--ref", type=str, nargs='+', required = False, help="list of reference images/videos")
    parser.add_argument("--gpu", type=int,  default=0, help="select which GPU to use (e.g. 0), default is GPU 0. Pass -1 to run on the CPU.")
    parser.add_argument("--heatmap", type=str, default="none", help="type of difference map (none, raw, threshold, supra-threshold)")
    parser.add_argument("--heatmap-dir", type=str, default=None, help="in which directory heatmaps should be stored (the default is the current directory)")
    parser.add_argument("--foveated", action='store_true', default=False, help="Run in a foveated mode (non-foveated is the default)")
    parser.add_argument("--display", type=str, default="standard_4k", help="display name, e.g. 'HTC Vive', or ? to print the list of models.")
    parser.add_argument("--display-models", type=str, default=None, help="A path to the JSON file with a list of display models")
    parser.add_argument("--nframes", type=int, default=-1, help="the number of video frames you want to compare")
    parser.add_argument("--quiet", action='store_true', default=False, help="Do not print any information but the final JOD value. Warning message will be still printed.")
    parser.add_argument("--verbose", action='store_true', default=False, help="Print out extra information.")
    parser.add_argument("--full-screen-resize", choices=['fast_bilinear', 'bilinear', 'bicubic', 'lanczos'], help="Both test and reference videos will be resized to match the full resolution of the display. Currently works only with videos.")
    parser.add_argument("--metrics", choices=['fvvdp', 'pu-psnr'], nargs='+', default=['fvvdp'], help='Select which metric(s) to run')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.quiet:
        log_level = logging.WARNING
    else:        
        log_level = logging.DEBUG if args.verbose else logging.INFO
        
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=log_level)

    if args.display == "?":
        fvvdp_display_photometry.list_displays(args.display_models)
        return

    if args.test is None or args.ref is None:
        logging.error( "Paths to both test and reference content needs to be specified.")
        return

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    logging.info("Running on device: " + str(device))

    # heatmap_types = {
    #     "threshold"   : {"scale" : 1.000, "colormap_type": "trichromatic"},
    #     "supra-threshold" : {"scale" : 0.333, "colormap_type": "dichromatic"},
    # }
    heatmap_types = ["raw", "threshold", "supra-threshold"]

    if args.heatmap == "none":
        args.heatmap = None

    if args.heatmap:
        if not args.heatmap in heatmap_types:
            logging.error( 'The recognized heatmap types are: "none", "raw", "threshold" and "supra-threshold"' )
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

    metrics = []
    display_photometry = None
    display_geometry = None
    for mm in args.metrics:
        if mm == 'fvvdp':
            fv = pyfvvdp.fvvdp( display_name=args.display, foveated=args.foveated, heatmap=args.heatmap, device=device, display_models=args.display_models )
            metrics.append( fv )
            display_photometry = fv.display_photometry
            display_geometry = fv.display_geometry
        elif mm == 'pu-psnr':
            if args.heatmap:
                logging.warning( f'Skipping heatmap as it is not supported by {mm}' )
            if args.foveated:
                logging.warning( f'Foveated mode is not supported by {mm}' )
            metrics.append( pyfvvdp.pu_psnr(device=device) )
        else:
            raise RuntimeError( f"Unknown metric {mm}")

        info_str = metrics[-1].get_info_string()
        if not info_str is None:
            logging.info( 'When reporting metric results, please include the following information:' )
            logging.info( info_str )

    # If none of the metrics requires display geometry/photometry, we still need those for video source
    if display_geometry is None:
        display_geometry = fvvdp_display_geometry.load(args.display, models_file=args.display_models)
    if display_photometry is None:
        display_photometry = fvvdp_display_photometry.load(args.display, models_file=args.display_models)

    if args.verbose:
        display_photometry.print()

    for kk in range( max(N_test, N_ref) ): # For each test and reference pair
        test_file = args.test[min(kk,N_test-1)]
        ref_file = args.ref[min(kk,N_ref-1)]
        logging.info(f"Predicting the quality of '{test_file}' compared to '{ref_file}'")
        for mm in metrics:
            vs = pyfvvdp.fvvdp_video_source_file( test_file, ref_file, display_photometry=display_photometry, full_screen_resize=args.full_screen_resize, resize_resolution=display_geometry.resolution, frames=args.nframes )
            Q_pred, stats = mm.predict_video_source(vs)
            if args.quiet:
                print( "{Q:0.4f}".format(Q=Q_pred) )
            else:
                units_str = f" [{mm.quality_unit()}]"
                print( "{met_name}={Q:0.4f}{units}".format(met_name=mm.short_name(), Q=Q_pred, units=units_str) )

            if do_heatmap and not stats is None:
                # diff_type = heatmap_types[args.heatmap]
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
