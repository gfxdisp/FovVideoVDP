

import os, sys
import argparse
from natsort import natsorted
import glob
import ffmpeg
import numpy as np
from PIL import Image
from fvvdp import FovVideoVDP, DisplayModel
from visualize_diff_map import visualize_diff_map
import torch
from pytorch_msssim import SSIM
from utils import *

def sanitize_filelist(filestrs, do_sort=True):
    files = []
    for filestr in filestrs:
        if "*" in filestr:
            curlist = glob.glob(filestr)
            files = files + curlist
        else:
            files.append(filestr)
    if do_sort:
        return natsorted(files)
    else:
        return files

def make_grayscale(x):
    # layout: NC***, channels R, G, B
    return (
        0.2126 * x[:,0:1,...] + 
        0.7152 * x[:,1:2,...] +
        0.0722 * x[:,2:3,...]
    )

def load_video_as_tensor(vidfile, device, frames=60):
    probe = ffmpeg.probe(vidfile)

    # select the first video stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    avg_fps_num, avg_fps_denom = [float(x) for x in video_stream['r_frame_rate'].split("/")]
    avg_fps = avg_fps_num/avg_fps_denom
    assert num_frames >= frames

    video = ffmpeg.input(vidfile)
    out, _ = (
        video.video.trim(start_frame=0, end_frame=frames).setpts('PTS-STARTPTS')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    npvideo = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )
    np_raw_tensor = npvideo.astype('float32') * 1.0/255.0
    torch_raw_tensor = torch.tensor(np_raw_tensor, device=device)
    # torchlinear = srgb2linear_torch(torch.tensor(npsrgb).to(device))
    tensor = torch_raw_tensor.permute((3, 0, 1, 2)).unsqueeze(dim=0)

    # if return_grayscale:
    #     tensor = make_grayscale(tensor)

    return tensor, avg_fps

def load_image_as_tensor(imgfile, device, frames=60):

    # # load image as srgb and convert to tensor
    # srgb_tensor = torch.tensor(img2np(Image.open(imgfile).convert("RGB"))).to(device)
    # lin_tensor = srgb2linear_torch(srgb_tensor)


    # if return_grayscale:
    #     lin_tensor = make_grayscale(lin_tensor)
    raw_tensor = torch.tensor(img2np(Image.open(imgfile).convert("RGB"))).to(device)

    # # add batch and frame dimensions
    raw_tensor = raw_tensor.permute(2, 0, 1).unsqueeze(dim=0).unsqueeze(dim=2)

    # repeat frames
    video_tensor = raw_tensor.expand(1, raw_tensor.shape[1], frames, raw_tensor.shape[3], raw_tensor.shape[4])
    #video_tensor = raw_tensor.expand(1, raw_tensor.shape[0], frames, raw_tensor.shape[1], raw_tensor.shape[2])

    return video_tensor, 60.0

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


# -----------------------------------
# Command-line Arguments
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FovVideoVDP on a set of videos")
    parser.add_argument ("--ref", type=str, required = True, help="ref image or video")
    parser.add_argument ("--test", type=str, nargs='+', required = True, help="list of test images/videos")
    parser.add_argument("--gpu", type=int,  default=-1, help="select which GPU to use (e.g. 0), default is CPU")
    parser.add_argument("--diff", type=str, default=None, help="type of difference map (None, subthreshold, suprathreshold)")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose mode")
    parser.add_argument("--display", type=str, default="24-inch SDR Monitor", help="display name, e.g. HTC Vive")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    print("Running on " + str(device))

    diff_types = {
        "subthreshold"   : {"scale" : 1.000, "colormap_type": "trichromatic"},
        "suprathreshold" : {"scale" : 0.333, "colormap_type": "dichromatic"},
    }

    if args.diff in diff_types:
        do_diff = True
    else:
        do_diff = False

    # these extensions switch mode to images instead
    image_extensions = [".png", ".jpg", ".gif", ".bmp", ".jpeg", ".ppm", ".tiff", ".dds"]

    if os.path.splitext(args.ref)[1] in image_extensions:
        loader = load_image_as_tensor
        print("Mode: Image")
    else:
        loader = load_video_as_tensor
        print("Mode: Video")

    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)

    args.test = sanitize_filelist(args.test, do_sort=False)

    display_model = DisplayModel.load(args.display, 'sRGB')

    ref_vid, ref_avg_fps = loader(args.ref, device=device)
    ref_vid_luminance = display_model.get_luminance_pytorch(ref_vid)

    H, W = ref_vid.shape[-2], ref_vid.shape[-1]

    vdploss = FovVideoVDP(H=H, W=W, display_model=display_model, frames_per_s=ref_avg_fps, do_diff_map=do_diff, device=device)

    for testfile in args.test:
        print(testfile + "... ", flush=True)

        test_vid, test_avg_fps = loader(testfile, device=device)

        if ref_avg_fps == test_avg_fps:
            cur_fps = ref_avg_fps
            cur_frames = min(ref_vid.shape[2], test_vid.shape[2])

            ssim = sum([ssim_module(ref_vid[:,:,i,...], test_vid[:,:,i,...]) for i in range(cur_frames)]) * 1.0/float(cur_frames)
            print("    SSIM %0.4f " % (ssim), end='', flush=True)

            loss, jod, diff_map = vdploss(display_model.get_luminance_pytorch(test_vid[:,:,0:cur_frames,...]), ref_vid_luminance[:,:,0:cur_frames,...])
            print( "VDP: %0.4f (JOD % 0.4f)" % (loss.cpu().item(), jod.cpu().item()))
        else:
            if ref_avg_fps.is_integer() and test_avg_fps.is_integer():
                cur_fps = np.lcm(int(ref_avg_fps), int(test_avg_fps))                
                print("    Upsampling ref (%0.2f) and test (%0.2f) to %d fps" % (ref_avg_fps, test_avg_fps, cur_fps))

                cur_ref_vid  = torch.repeat_interleave(ref_vid,  cur_fps//int(ref_avg_fps),  axis=2)
                cur_test_vid = torch.repeat_interleave(test_vid, cur_fps//int(test_avg_fps), axis=2)

                cur_frames = min(120, min(cur_ref_vid.shape[2], cur_test_vid.shape[2]))

                # np2vid((linear2srgb_torch(cur_ref_vid[0])).permute((1,2,3,0)).cpu().numpy(), "cur_ref.mp4", cur_fps)
                # np2vid((linear2srgb_torch(cur_test_vid[0])).permute((1,2,3,0)).cpu().numpy(), "cur_test.mp4", cur_fps)

                cur_ref_vid  = cur_ref_vid [:,:,0:cur_frames,...]
                cur_test_vid = cur_test_vid[:,:,0:cur_frames,...]

                if cur_frames == 120:
                    print("Limiting to 120 frames")

                cur_vdploss = FovVideoVDP(H=H, W=W, display_model=display_model, frames_per_s=cur_fps, do_diff_map=do_diff, device=device)

                ssim = sum([ssim_module(cur_ref_vid[:,:,i,...], cur_test_vid[:,:,i,...]) for i in range(cur_frames)]) * 1.0/float(cur_frames)
                print("    SSIM %0.4f " % (ssim), end='', flush=True)

                loss, jod, diff_map = cur_vdploss(display_model.get_luminance_pytorch(cur_test_vid), display_model.get_luminance_pytorch(cur_ref_vid))
                print( "VDP: %0.4f (JOD % 0.4f)" % (loss.cpu().item(), jod.cpu().item()))
            else:
                print("    Error: Ref (%0.2f) and test (%0.2f) videos have different fps, and not integers" % (ref_avg_fps, test_avg_fps))
                continue

        if do_diff:
            diff_type = diff_types[args.diff]
            print("    Writing diff map...")
            diff_map = diff_map * diff_type["scale"]
            diff_map_viz = visualize_diff_map(diff_map, context_image=ref_vid, colormap_type=diff_type["colormap_type"])
            out_dir = os.path.join(os.path.dirname(testfile), "diff_maps")
            os.makedirs(out_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(testfile))
            np2vid((linear2srgb_torch(diff_map[0])).permute((1,2,3,0)).cpu().numpy(), os.path.join(out_dir, base + "_diff_map.mp4"), cur_fps, args.verbose)
            np2vid((diff_map_viz[0]).permute((1,2,3,0)).cpu().numpy(), os.path.join(out_dir, base + "_diff_map_viz.mp4"), cur_fps, args.verbose)

            del diff_map
            del diff_map_viz

        del test_vid
        torch.cuda.empty_cache()
