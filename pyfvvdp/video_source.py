from abc import abstractmethod
import torch
import os
import numpy as np 
from torch.functional import Tensor
import pyfvvdp.utils as utils
from pyfvvdp.fvvdp_display_model import fvvdp_display_photometry, fvvdp_display_geometry

"""
fvvdp_video_source_* objects are used to supply test/reference frames to FovVideoVDP. 
Those could be comming from memory or files. The subclasses of this abstract class implement
reading the frames and converting them to the approprtate format. 
"""
class fvvdp_video_source:
   
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    @abstractmethod
    def get_video_size(self):
        pass

    # Return the frame rate of the video
    @abstractmethod
    def get_frames_per_second(self) -> int:
        pass
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    @abstractmethod
    def get_test_frame( self, frame, device ) -> Tensor:
        pass

    @abstractmethod
    def get_reference_frame( self, frame, device ) -> Tensor:
        pass


"""
Function for changing the order of dimensions, for example, from "WHC" (width, height, colour) to "BCHW" (batch, colour, height, width)
If a dimension is missing in in_dims, it will be added as a singleton dimension
"""
def reshuffle_dims( T: Tensor, in_dims: str, out_dims: str ) -> Tensor:
    in_dims = in_dims.upper()
    out_dims = out_dims.upper()    

    # Find intersection of two strings    
    inter_dims = ""
    for kk in range(len(out_dims)):
        if in_dims.find(out_dims[kk]) != -1:
            inter_dims += out_dims[kk]

    # First, permute into the right order
    perm = [0] * len(inter_dims)
    for kk in range(len(inter_dims)):
        ind = in_dims.find(inter_dims[kk])
        if ind == -1:
            raise RuntimeError( 'Dimension "{}" missing in the target dimensions: "{}"'.format(in_dims[kk],out_dims) )
        perm[kk] = ind                    
    T_p = T.permute(perm)

    # Add missing dimensions
    out_sh = [1] * len(out_dims)
    for kk in range(len(out_dims)):        
        ind = inter_dims.find(out_dims[kk])
        if ind != -1:
            out_sh[kk] = T_p.shape[ind]

    return T_p.reshape( out_sh )


"""
This video_source uses a photometric display model to convert input content (e.g. sRGB) to luminance maps. 
"""
class fvvdp_video_source_dm( fvvdp_video_source ):

    def __init__( self,  display_photometry='sdr_4k_30', color_space_name='sRGB' ):

        #colorspaces_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/color_spaces.json")
        colorspaces_file = utils.config_files.find( "color_spaces.json" )
        colorspaces = utils.json2dict(colorspaces_file)

        if not color_space_name in colorspaces:
            raise RuntimeError( "Unknown color space: \"" + color_space_name + "\"" )

        self.color_to_luminance = colorspaces[color_space_name]['RGB2Y']

        if isinstance( display_photometry, str ):
            self.dm_photometry = fvvdp_display_photometry.load(display_photometry) 
        elif isinstance( display_photometry, fvvdp_display_photometry ):
            self.dm_photometry = display_photometry
        else:
            raise RuntimeError( "display_model must be a string or fvvdp_display_photometry subclass" )


"""
This video source supplies frames from either Pytorch tensors and Numpy arrays. It also applies a photometric display model.

A batch of videos should be stored as a tensor or numpy array. Ideally, the tensor should have the dimensions BCFHW (batch, colour, frame, height, width).If tensor is stored in another formay, you can pass the order of dimsions as "dim_order" parameter. If any dimension is missing, it will
be added as a singleton dimension. 

This class is for display-encoded (gamma-encoded) content that will be processed by a display model to produce linear  absolute luminance emitted from a display.
"""
class fvvdp_video_source_array( fvvdp_video_source_dm ):
               
    # test_video, reference video - tensor with test and reference video frames. See the class description above for the explanation of dimensions of those tensors.
    # fps - frames per second. Must be 0 for images
    # dim_order - a string with the order of the dimensions. 'BCFHW' is the default.
    # display_model - object that implements fvvdp_display_photometry
    #   class
    # color_space_name - name of the colour space (see
    #   fvvdp_data/color_spaces.json)
    def __init__( self, test_video, reference_video, fps, dim_order='BCFHW', display_photometry='sdr_4k_30', color_space_name='sRGB' ):

        super().__init__(display_photometry=display_photometry, color_space_name=color_space_name)        

        if test_video.shape != reference_video.shape:
            raise RuntimeError( 'Test and reference image/video tensors must be exactly the same shape' )
        
        if len(dim_order) != len(test_video.shape):
            raise RuntimeError( 'Input tensor much have exactly as many dimensions as there are characters in the "dims" parameter' )

        # Convert numpy arrays to tensors. Note that we do not upload to device or change dtype at this point (to save GPU memory)
        if isinstance( test_video, np.ndarray ):
            if test_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Test has datatype uint16, packing into int16')
                test_video = test_video.astype(np.int16)
            test_video = torch.tensor(test_video)
        if isinstance( reference_video, np.ndarray ):
            if reference_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Reference has datatype uint16, packing into int16')
                reference_video = reference_video.astype(np.int16)
            reference_video = torch.tensor(reference_video)

        # Change the order of dimension to match BFCHW - batch, frame, colour, height, width
        test_video = reshuffle_dims( test_video, in_dims=dim_order, out_dims="BCFHW" )
        reference_video = reshuffle_dims( reference_video, in_dims=dim_order, out_dims="BCFHW" )

        B, C, F, H, W = test_video.shape
        
        if fps==0 and F>1:
            raise RuntimeError( 'When passing video sequences, you must set ''frames_per_second'' parameter' )

        if C!=3 and C!=1:
            raise RuntimeError( 'The content must have either 1 or 3 colour channels.' )

        self.fps = fps
        self.is_video = (fps>0)
        self.is_color = (C==3)
        self.test_video = test_video
        self.reference_video = reference_video


    def get_frames_per_second(self):
        return self.fps
            
    # Return a [height width frames] vector with the resolution and
    # the number of frames in the video clip. [height width 1] is
    # returned for an image. 
    def get_video_size(self):

        sh = self.test_video.shape
        return (sh[3], sh[4], sh[2])
    
    # % Get a test video frame as a single-precision luminance map
    # % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # % starting from 0. If use_gpu==true, the function should return a
    # % gpuArray.

    def get_test_frame( self, frame, device=torch.device('cpu') ):
        return self._get_frame(self.test_video, frame, device )

    def get_reference_frame( self, frame, device=torch.device('cpu') ):
        return self._get_frame(self.reference_video, frame, device )

    def _get_frame( self, from_array, frame, device ):        
        # Determine the maximum value of the data type storing the
        # image/video

        if from_array.dtype in (torch.float32, torch.float16):
            frame = from_array[:,:,frame:(frame+1),:,:].to(device)
            if from_array.dtype is torch.float16 and device == torch.device('cpu'):
                # Torch CPU does not support some operations for float16
                frame = frame.to(torch.float32)
        elif from_array.dtype is torch.int16:
            # Use int16 to losslessly pack uint16 values
            # Unpack from int16 by bit masking as described in this thread:
            # https://stackoverflow.com/a/20766900
            # logging.info('Found int16 datatype, unpack into uint16')
            max_value = 2**16 - 1
            # Cast to int32 to store values >= 2**15
            frame_int32 = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.int32)
            frame_uint16 = frame_int32 & max_value
            # Finally convert to float in the range [0,1]
            frame = frame_uint16.to(torch.float32) / max_value
        elif from_array.dtype is torch.uint8:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.float32)/255
        else:
            raise RuntimeError( "Only uint8, uint16 and float32 is currently supported" )

        L = self.dm_photometry.forward( frame )
        
        if self.is_color:
            # Convert to grayscale
            L = L[:,0:1,:,:,:]*self.color_to_luminance[0] + L[:,1:2,:,:,:]*self.color_to_luminance[1] + L[:,2:3,:,:,:]*self.color_to_luminance[2]

        return L
