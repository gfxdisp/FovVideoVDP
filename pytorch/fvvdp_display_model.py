import torch
import torch.nn.functional as Func
import numpy as np 
import math
import logging

# Convert pixel values to linear RGB using sRGB non-linearity
# 
# L = srgb2lin( p )
#
# p - pixel values (between 0 and 1)
# L - relative linear RGB (or luminance), normalized to the range 0-1
def srgb2lin( p ):
    L = torch.where(p > 0.04045, ((p + 0.055) / 1.055)**2.4, p/12.92)
    return L


class fvvdp_display_photo_gog: 
    # Gain-gamma-offset display model to simulate SDR displays
    # Object variables:
    #  Y_peak - display peak luminance in cd/m^2
    #  contrast - display contrast 
    #  gamma
    #  E_ambient
    #  k_refl
    
    # Gain-gamma-offset display model to simulate SDR displays
    #
    # dm = fvvdp_display_photo_gog( Y_peak )
    # dm = fvvdp_display_photo_gog( Y_peak, contrast )
    # dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma )
    # dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient )
    # dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient, k_refl )
    #
    # Parameters (default value shown in []):
    # Y_peak - display peak luminance in cd/m^2 (nit), e.g. 200 for a typical
    #          office monitor
    # contrast - [1000] the contrast of the display. The value 1000 means
    #          1000:1
    # gamma - [-1] gamma of the display, typically 2.2. If -1 is
    #         passed, sRGB non-linearity is used.         
    # E_ambient - [0] ambient light illuminance in lux, e.g. 600 for bright
    #         office
    # k_refl - [0.005] reflectivity of the display screen
    #
    # For more details on the GOG display model, see:
    # https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016perceptual_display.pdf
    #
    # Copyright (c) 2010-2021, Rafal Mantiuk
    def __init__( self, Y_peak, contrast = 1000, gamma = -1, E_ambient = 0, k_refl = 0.005 ):
            
        self.Y_peak = Y_peak            
        self.contrast = contrast
        self.gamma = gamma
        self.E_ambient = E_ambient
        self.k_refl = k_refl
    
        
    # Transforms gamma-correctec pixel values V, which must be in the range
    # 0-1, into absolute linear colorimetric values emitted from
    # the display.
    def forward( self, V ):
        
        if torch.any(V>1).bool() or torch.any(V<0).bool():
            logging.warning("Pixel outside the valid range 0-1")
            V = V.clamp( 0., 1. )
            
        Y_black = self.get_black_level()
        
        if self.gamma==-1: # sRGB
            L = (self.Y_peak-Y_black)*srgb2lin(V) + Y_black
        else:
            L = (self.Y_peak-Y_black)*torch.pow(V, self.gamma) + Y_black
        
        return L
        

    def  get_peak_luminance( self ):
        return self.Y_peak


    def get_black_level( self ):
        Y_refl = self.E_ambient/math.pi*self.k_refl  # Reflected ambient light            
        Y_black = Y_refl + self.Y_peak/self.contrast

        return Y_black
    
    def print( self ):
        Y_black = self.get_black_level()
        
        logging.info( 'Photometric display model:' )
        logging.info( '  Peak luminance: {} cd/m^2'.format(self.Y_peak) )
        logging.info( '  Contrast - theoretical: {}:1'.format( math.round(self.contrast) ) )
        logging.info( '  Contrast - effective: {}:1'.format, math.round(self.Y_peak/Y_black) )
        logging.info( '  Ambient light: {} lux'.format( self.E_ambient ) )
        logging.info( '  Display reflectivity: {}%'.format( self.k_refl*100 ) )
    

