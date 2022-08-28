# Classes for reading images or videos from files so that they can be passed to FovVideoVDP frame-by-frame

import os
import imageio.v2 as io
import numpy as np
from torch.functional import Tensor
import torch
import ffmpeg

from pyfvvdp.fvvdp import fvvdp_video_source, fvvdp_video_source_dm, fvvdp_video_source_array, reshuffle_dims

def load_image_as_array(imgfile):
    # 16-bit PNG not supported by default
    lib = 'PNG-FI' if os.path.splitext(imgfile)[1].lower() == '.png' else None
    # If the line below fails on Windows stating that "it cannot `"PNG-FI` can not handle the given uri.", 
    # run `imageio.plugins.freeimage.download()` or check imageio.help( 'PNG-FI' )
    img = io.imread(imgfile, format=lib)
    return img


class video_reader:

    def __init__(self, vidfile, frames):
        try:
            probe = ffmpeg.probe(vidfile)
        except:
            raise RuntimeError("ffmpeg failed to open file \"" + vidfile + "\"")

        # select the first video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])
        num_frames = int(video_stream['nb_frames'])
        avg_fps_num, avg_fps_denom = [float(x) for x in video_stream['r_frame_rate'].split("/")]
        self.avg_fps = avg_fps_num/avg_fps_denom

        if frames==-1:
            self.frames = num_frames
        else:    
            if num_frames < frames:
                err_str = 'Expecting {needed_frames} frames but only {available_frames} available in the file \"{file}\"'.format( needed_frames=frames, available_frames=num_frames, file=vidfile)
                raise RuntimeError( err_str )
            self.frames = frames

        self.process = (
            ffmpeg
            .input(vidfile)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
        )

        self.curr_frame = -1

    def get_frame(self):
        in_bytes = self.process.stdout.read(self.width * self.height * 3)
        if not in_bytes or self.curr_frame == self.frames:
            return None
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([self.height, self.width, 3])
        ) 
        self.curr_frame += 1
        return in_frame       


'''
Use ffmpeg to read video frames, one by one.
'''
class fvvdp_video_source_video_file(fvvdp_video_source_dm):

    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', color_space_name='sRGB', frames=-1 ):

        super().__init__(display_photometry=display_photometry, color_space_name=color_space_name)        

        self.test_vidr = video_reader(test_fname, frames)
        self.reference_vidr = video_reader(reference_fname, frames)
        if frames==-1:
            self.frames = self.test_vidr.frames
        else:
            self.frames = frames

        if self.test_vidr.height != self.reference_vidr.height or self.test_vidr.width != self.reference_vidr.width:
            raise RuntimeError( 'Test and reference video sequences must have the same resolutions. Found: test {tw}x{th}, reference {rw}x{rh}'.format(tw=self.test_vidr.width, th=self.reference_vidr.height, rw=self.test_vidr.width, rh=self.referemce_vidr.height) )

        # self.last_test_frame = None
        # self.last_reference_frame = None
        
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        return (self.test_vidr.height, self.test_vidr.width, self.frames )

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.test_vidr.avg_fps
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    def get_test_frame( self, frame, device ) -> Tensor:
        # if not self.last_test_frame is None and frame == self.last_test_frame[0]:
        #     return self.last_test_frame[1]
        L = self._get_frame( self.test_vidr, frame, device )
        # self.last_test_frame = (frame,L)
        return L

    def get_reference_frame( self, frame, device ) -> Tensor:
        # if not self.last_reference_frame is None and frame == self.last_reference_frame[0]:
        #     return self.last_reference_frame[1]
        L = self._get_frame( self.reference_vidr, frame, device )
        # self.reference_test_frame = (frame,L)
        return L

    def _get_frame( self, vid_reader, frame, device ):        

        if frame != (vid_reader.curr_frame+1):
            raise RuntimeError( 'Video can be currently only read frame-by-frame. Random access not implemented.' )

        frame_np = vid_reader.get_frame()

        if frame_np is None:
            raise RuntimeError( 'Could not read frame {}'.format(frame) )

        frame_t_hwc = torch.tensor(frame_np)
        assert frame_t_hwc.dtype is torch.uint8
        frame_t = reshuffle_dims( frame_t_hwc, in_dims='HWC', out_dims="BCFHW" )
        frame_t = frame_t.to(device).to(torch.float32)/255

        L = self.dm_photometry.forward( frame_t )

        # Convert to grayscale
        L = L[:,0:1,:,:,:]*self.color_to_luminance[0] + L[:,1:2,:,:,:]*self.color_to_luminance[1] + L[:,2:3,:,:,:]*self.color_to_luminance[2]

        return L

'''
Recognize whether the file is an image of video and wraps an appropriate video_source for the given content.
'''
class fvvdp_video_source_file(fvvdp_video_source):

    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', color_space_name='sRGB', frames=-1 ):
        # these extensions switch mode to images instead
        image_extensions = [".png", ".jpg", ".gif", ".bmp", ".jpeg", ".ppm", ".tiff", ".dds", ".exr", ".hdr"]

        assert os.path.isfile(test_fname), f'File does not exists: "{test_fname}"'
        assert os.path.isfile(reference_fname), f'File does not exists: "{reference_fname}"'

        if os.path.splitext(test_fname)[1].lower() in image_extensions:
            assert os.path.splitext(reference_fname)[1].lower() in image_extensions, 'Test is an image, but reference is a video'
            img_test = load_image_as_array(test_fname)
            img_reference = load_image_as_array(reference_fname)
            self.vs = fvvdp_video_source_array( img_test, img_reference, 0, dim_order='HWC', display_photometry=display_photometry, color_space_name=color_space_name )
        else:
            assert os.path.splitext(reference_fname)[1].lower() not in image_extensions, 'Test is a video, but reference is an image'
            self.vs = fvvdp_video_source_video_file( test_fname, reference_fname, display_photometry=display_photometry, color_space_name=color_space_name, frames=frames )


    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        return self.vs.get_video_size()

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.vs.get_frames_per_second()
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    def get_test_frame( self, frame, device ) -> Tensor:
        return self.vs.get_test_frame( frame, device )

    def get_reference_frame( self, frame, device ) -> Tensor:
        return self.vs.get_reference_frame( frame, device )
