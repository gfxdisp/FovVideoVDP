# Classes for reading images or videos from files so that they can be passed to FovVideoVDP frame-by-frame

import os
import imageio.v2 as io
import numpy as np
from torch.functional import Tensor
import torch
import ffmpeg

from pyfvvdp.fvvdp import fvvdp_video_source, fvvdp_video_source_dm, fvvdp_video_source_array, reshuffle_dims

# for debugging only
from gfxdisp.pfs import pfs

try:
    # This may fail if OpenEXR is not installed. To install,
    # ubuntu: sudo apt install libopenexr-dev
    # mac: brew install openexr
    import pyexr
    use_pyexr = True
except ImportError as e:
    # Imageio's imread is unreliable for OpenEXR images
    # See https://github.com/imageio/imageio/issues/517
    use_pyexr = False

def load_image_as_array(imgfile):
    ext = os.path.splitext(imgfile)[1].lower()
    if ext == '.exr' and use_pyexr:
        precisions = pyexr.open(imgfile).precisions
        assert precisions.count(precisions[0]) == len(precisions), 'All channels must have same precision'
        img = pyexr.read(imgfile, precision=precisions[0])
    else:
        # 16-bit PNG not supported by default
        lib = 'PNG-FI' if ext == '.png' else None
        img = io.imread(imgfile, format=lib)
    return img


class video_reader:

    def __init__(self, vidfile, frames, resize_fn=None, resize_height=-1, resize_width=-1):
        try:
            probe = ffmpeg.probe(vidfile)
        except:
            raise RuntimeError("ffmpeg failed to open file \"" + vidfile + "\"")

        # select the first video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])
        self.color_space = video_stream['color_space']
        self.color_transfer = video_stream['color_transfer']
        in_pix_fmt=video_stream['pix_fmt']
        if ('p10' in in_pix_fmt) or ('p12' in in_pix_fmt) or ('p14' in in_pix_fmt) or ('p16' in in_pix_fmt): # >8 bit
            self.out_pix_fmt='rgb48le'
            self.bpp = 6 # bytes per pixel
            self.dtype = np.uint16
        else:
            self.out_pix_fmt='rgb24' # 8 bit
            self.bpp = 6 # bytes per pixel
            self.dtype = np.uint8

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

        # self.process = (
        #     ffmpeg
        #     .input(vidfile)
        #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        #     .run_async(pipe_stdout=True, quiet=True)
        # )
        stream = ffmpeg.input(vidfile)
        if (resize_fn is not None) and (resize_width!=self.width or resize_height!=self.height):
            stream = ffmpeg.filter(stream, 'scale', resize_width, resize_height, flags=resize_fn)
            self.width = resize_width
            self.height = resize_height
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt=self.out_pix_fmt)
        self.process = ffmpeg.run_async(stream, pipe_stderr=True, quiet=True)

        self.curr_frame = -1

    def get_frame(self):
        in_bytes = self.process.stdout.read(self.width * self.height * self.bpp )
        if not in_bytes or self.curr_frame == self.frames:
            return None
        in_frame = (
            np
            .frombuffer(in_bytes, self.dtype)
            .reshape([self.height, self.width, 3])
        ) 
        self.curr_frame += 1
        return in_frame       


'''
Use ffmpeg to read video frames, one by one.
'''
class fvvdp_video_source_video_file(fvvdp_video_source_dm):

    #   
    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', color_space_name='auto', frames=-1, display_models=None, full_screen_resize=None, resize_resolution=None ):

        fs_width = -1 if full_screen_resize is None else resize_resolution[0]
        fs_height = -1 if full_screen_resize is None else resize_resolution[1]
        self.reference_vidr = video_reader(reference_fname, frames, resize_fn=full_screen_resize, resize_width=fs_width, resize_height=fs_height)
        self.test_vidr = video_reader(test_fname, frames, resize_fn=full_screen_resize, resize_width=fs_width, resize_height=fs_height)

        if color_space_name=='auto':
            if self.test_vidr.color_space=='bt2020nc':
                color_space_name="BT.2020"
            else:
                color_space_name="sRGB"

        super().__init__(display_photometry=display_photometry, color_space_name=color_space_name, display_models=display_models)        

        if self.test_vidr.color_transfer=="smpte2084" and self.dm_photometry.EOTF!="PQ":
            raise RuntimeError( f"Video color transfer function ({self.test_vidr.color_transfer}) inconsistent with EOTF of the display model ({self.dm_photometry.EOTF})" )

        if frames==-1:
            self.frames = self.test_vidr.frames
        else:
            self.frames = frames

        if self.test_vidr.height != self.reference_vidr.height or self.test_vidr.width != self.reference_vidr.width:
            raise RuntimeError( f'Test and reference video sequences must have the same resolutions. Found: test {self.test_vidr.width}x{self.test_vidr.height}, reference {self.reference_vidr.width}x{self.reference_vidr.height}' )

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

        if frame_np.dtype == np.uint16:
            # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
            # This will be efficiently transferred and unpacked on the GPU.
            # logging.info('Test has datatype uint16, packing into int16')
            frame_t_hwc = torch.tensor(frame_np.astype(np.int16))
            assert frame_t_hwc.dtype is torch.int16
            frame_t = reshuffle_dims( frame_t_hwc, in_dims='HWC', out_dims="BCFHW" )
            max_value = 2**16 - 1
            frame_int32 = frame_t.to(device).to(torch.int32)
            frame_uint16 = frame_int32 & max_value
            frame_t = frame_uint16.to(torch.float32)/max_value
        else:
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

    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', color_space_name='auto', frames=-1, display_models=None, full_screen_resize=None, resize_resolution=None ):
        # these extensions switch mode to images instead
        image_extensions = [".png", ".jpg", ".gif", ".bmp", ".jpeg", ".ppm", ".tiff", ".dds", ".exr", ".hdr"]

        assert os.path.isfile(test_fname), f'File does not exists: "{test_fname}"'
        assert os.path.isfile(reference_fname), f'File does not exists: "{reference_fname}"'

        if os.path.splitext(test_fname)[1].lower() in image_extensions:
            assert os.path.splitext(reference_fname)[1].lower() in image_extensions, 'Test is an image, but reference is a video'
            if color_space_name=='auto':
                color_space_name='sRGB' # TODO: detect the right colour space
            img_test = load_image_as_array(test_fname)
            img_reference = load_image_as_array(reference_fname)
            self.vs = fvvdp_video_source_array( img_test, img_reference, 0, dim_order='HWC', display_photometry=display_photometry, color_space_name=color_space_name, display_models=display_models )
        else:
            assert os.path.splitext(reference_fname)[1].lower() not in image_extensions, 'Test is a video, but reference is an image'
            self.vs = fvvdp_video_source_video_file( test_fname, reference_fname, display_photometry=display_photometry, color_space_name=color_space_name, frames=frames, display_models=display_models, full_screen_resize=full_screen_resize, resize_resolution=resize_resolution )


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
