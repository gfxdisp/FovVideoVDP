from abc import ABC, abstractmethod
import torch
import torch.nn.functional as Func
import numpy as np 
import math
import logging
import os

from utils import json2dict

# Convert pixel values to linear RGB using sRGB non-linearity
# 
# L = srgb2lin( p )
#
# p - pixel values (between 0 and 1)
# L - relative linear RGB (or luminance), normalized to the range 0-1
def srgb2lin( p ):
    L = torch.where(p > 0.04045, ((p + 0.055) / 1.055)**2.4, p/12.92)
    return L

class fvvdp_display_photometry:

    # Transforms gamma-encoded pixel values V, which must be in the range
    # 0-into absolute linear colorimetric values emitted from
    # the display.
    @abstractmethod
    def forward( self, V ):
        pass

    # Print the display specification    
    @abstractmethod
    def print( self ):
        pass

    @classmethod
    def load( cls, display_name ):
        models_file      = os.path.join(os.path.dirname(__file__), "fvvdp_data/display_models.json")

        models = json2dict(models_file)

        if not display_name in models:
            raise RuntimeError( "Unknown display model: \"" + display_name + "\"" )

        model = models[display_name]

        Y_peak = model["max_luminance"]

        if "min_luminance" in model:
            contrast = Y_peak/model["min_luminance"]
        elif "contrast" in model:
            contrast = model["contrast"]
        else:
            contrast = 500

        # Ambient light
        if "E_ambient" in model:
            E_ambient = model["E_ambient"]
        else:
            E_ambient = 0
        
        # Reflectivity of the display panel
        if "k_refl" in model: 
            k_refl = model["k_refl"]
        else:
            k_refl = 0.005

        # Reflectivity of the display panel
        if "gamma" in model: 
            gamma = model["gamma"]
        else:
            gamma = 2.2

        obj = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient, k_refl)
        obj.full_name = model["name"]
        obj.short_name = display_name

        return obj




class fvvdp_display_photo_gog(fvvdp_display_photometry): 
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
    def __init__( self, Y_peak, contrast = 1000, gamma = 2.2, E_ambient = 0, k_refl = 0.005 ):
            
        self.Y_peak = Y_peak            
        self.contrast = contrast
        self.gamma = gamma
        self.E_ambient = E_ambient
        self.k_refl = k_refl
    
        
    # Transforms gamma-encoded pixel values V, which must be in the range
    # 0-into absolute linear colorimetric values emitted from
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
        

    def get_peak_luminance( self ):
        return self.Y_peak


    # Get the effective black level, accounting for screen reflections
    def get_black_level( self ):
        Y_refl = self.E_ambient/math.pi*self.k_refl  # Reflected ambient light            
        Y_black = Y_refl + self.Y_peak/self.contrast

        return Y_black

    # Print the display specification    
    def print( self ):
        Y_black = self.get_black_level()
        
        logging.info( 'Photometric display model:' )
        logging.info( '  Peak luminance: {} cd/m^2'.format(self.Y_peak) )
        logging.info( '  Contrast - theoretical: {}:1'.format( math.round(self.contrast) ) )
        logging.info( '  Contrast - effective: {}:1'.format, math.round(self.Y_peak/Y_black) )
        logging.info( '  Ambient light: {} lux'.format( self.E_ambient ) )
        logging.info( '  Display reflectivity: {}%'.format( self.k_refl*100 ) )
    

class fvvdp_display_photo_absolute(fvvdp_display_photometry):
    # Use this photometric model when passing absolute colorimetric of
    # photometric values, scaled in cd/m^2
    # Object variables:
    #  L_max - display peak luminance in cd/m^2
    #  L_min - display black level
    def __init__(self, L_max=10000, L_min=0.005):

        self.L_max = L_max
        self.L_min = L_min


    def forward( self, V ):

        # Clamp the values that are outside the (L_min, L_max) range.
        L = V.clamp(self.L_min, self.L_max)

        if V.max() < 1:
            logging.warning('Pixel values are very low. Perhaps images are' \
                            ' not scaled in the absolute units of cd/m^2.')

        return L


    def  get_peak_luminance( self ):
        return self.L_max


    def get_black_level( self ):
        return self.L_min

    # Print the display specification
    def print( self ):
        Y_black = self.get_black_level()

        logging.info('Photometric display model:')
        logging.info('  Absolute photometric/colorimetric values')


# Use this class to compute the effective resolution of a display in pixels
# per degree (ppd). The class accounts for the change in the projection
# when looking at large FOV displays (e.g. VR headsets) at certain
# eccentricity.
#
# The class is also useful for computing the size of a display in meters
# and visual degrees. Check 'display_size_m' and 'display_size_deg' class
# properties for that.
#
# R = fvvdp_display_geometry(resolution, distance_m=None, distance_display_heights=None, 
#                            fov_horizontal=None, fov_vertical=None, fov_diagonal=None, 
#                            diagonal_size_inches=None)
#
# resolution is the 2-element touple with the pixel resolution of the
# display: (horizontal_resolutution, vertical_resolutution)
# distance_m - viewing distance in meters
# distance_display_heights - viewing distance in the heights of a display
# fov_horizontal - horizontal field of view of the display in degrees
# fov_vertical - vertical field of view of the display in degrees
# fov_diagonal - diagonal field of view of the display in degrees
# diagonal_size_inches - display diagonal resolution in inches
#
# Examples:
# # HTC Pro
# # Note that the viewing distance must be specified even though the resolution
# # and 'fov_diagonal' are enough to find pix_per_deg.
# R = fvvdp_display_geometry( (1440, 1600), distance_m=3, fov_diagonal=110 )
# R.get_ppd( torch.tensor( [0, 10, 20, 30, 40])) # pix per deg at the given eccentricities
#
# # 30" 4K monitor seen from 0.6 meters
# R = fvvdp_display_geometry( (3840, 2160), diagonal_size_inches=30, distance_m=0.6 )
# R.get_ppd()
#
# # 47" SIM2 display seen from 3 display heights
# R = fvvdp_display_geometry( (1920, 1080), diagonal_size_inches=47, distance_display_heights=3 )
# R.get_ppd()
#
# Some information about the effective FOV of VR headsets
# http://www.sitesinvr.com/viewer/htcvive/index.html
class fvvdp_display_geometry:

    def __init__(self, resolution, distance_m=None, distance_display_heights=None, fov_horizontal=None, fov_vertical=None, fov_diagonal=None, diagonal_size_inches=None) -> None:

        self.resolution = resolution
        
        ar = resolution[0]/resolution[1] # width/height
        
        self.fixed_ppd = None

        if not diagonal_size_inches is None:
            height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
            self.display_size_m = (ar*height_mm/1000, height_mm/1000)
                
        if (not distance_m is None) and (not distance_display_heights is None):
            raise RuntimeError( 'You can pass only one of: ''distance_m'', ''distance_display_heights''.' )
        
        if not distance_m is None:
            self.distance_m = distance_m;
        elif not distance_display_heights is None:
            if not hasattr( self, "display_size_m" ):
                raise RuntimeError( 'You need to specify display diagonal size ''diagonal_size_inches'' to specify viewing distance as ''distance_display_heights'' ' )
            self.distance_m = distance_display_heights * self.display_size_m[1]
        elif (not fov_horizontal is None) or (not fov_vertical is None) or (not fov_diagonal is None):
            # Default viewing distance for VR headsets
            self.distance_m = 3
        else:
            raise RuntimeError( 'Viewing distance must be specified as ''distance_m'' or ''distance_display_heights''.' )
        
        if ((not fov_horizontal is None) + (not fov_vertical is None) + (not fov_diagonal is None)) > 1:
            raise RuntimeError( 'You can pass only one of ''fov_horizontal'', ''fov_vertical'', ''fov_diagonal''. The other dimensions are inferred from the resolution assuming that the pixels are square.' )
        
        if not fov_horizontal is None:
            width_m = 2*math.tan( math.radians(fov_horizontal/2) )*self.distance_m
            self.display_size_m = (width_m, width_m/ar)
        elif not fov_vertical is None:
            height_m = 2*math.tan( math.radians(fov_vertical/2) )*self.distance_m
            self.display_size_m = (height_m*ar, height_m)
        elif not fov_diagonal is None:
            # Note that we cannot use Pythagorean theorem on degs -
            # we must operate on a distance measure
            # This is incorrect: height_deg = p.Results.fov_diagonal / sqrt( 1+ar^2 );
            
            distance_px = math.sqrt(self.resolution[0]**2 + self.resolution[1]**2) / (2.0 * math.tan( math.radians(fov_diagonal*0.5)) )
            height_deg = math.degrees(math.atan( self.resolution[1]/2 / distance_px ))*2
            
            height_m = 2*math.tan( math.radians(height_deg/2) )*self.distance_m
            self.display_size_m = (height_m*ar, height_m)
        
        self.display_size_deg = ( 2 * math.degrees(math.atan( self.display_size_m[0] / (2*self.distance_m) )), \
                                  2 * math.degrees(math.atan( self.display_size_m[1] / (2*self.distance_m) )) )
        
    # Get the number of pixels per degree
    #
    # ppd = R.get_ppd()
    # ppd = R.get_ppd(eccentricity)
    #
    # eccentricity is the viewing angle from the center in degrees. If
    # not specified, the central ppd value (for 0 eccentricity) is
    # returned.
    def get_ppd(self, eccentricity = None):
        
        # if ~isempty( dr.fixed_ppd )
        #     ppd = dr.fixed_ppd;
        #     return;
        # end
        
        # pixel size in the centre of the display
        pix_deg = 2*math.degrees(math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m ))
        
        base_ppd = 1/pix_deg
        
        if eccentricity is None:
            return base_ppd
        else:
            delta = pix_deg/2
            tan_delta = math.tan(math.radians(delta))
            tan_a = torch.tan( torch.deg2rad(eccentricity) )
            
            ppd = base_ppd * (torch.tan(torch.deg2rad(eccentricity+delta))-tan_a)/tan_delta
            return ppd


    # Convert pixel positions into eccentricities for the given
    # display
    #
    # resolution_pix - image resolution as [width height] in pix
    # x_pix, y_pix - pixel coordinates generated with meshgrid,
    #   pixels indexed from 0
    # gaze_pix - [x y] of the gaze position, in pixels
    def pix2eccentricity( self, resolution_pix, x_pix, y_pix, gaze_pix ):
                        
        if not self.fixed_ppd is None:
            ecc = torch.sqrt( (x_pix-gaze_pix[0])**2 + (y_pix-gaze_pix[1])**2 )/self.fixed_ppd
        else:
            # Position the image in the centre
            shift_to_centre = -resolution_pix/2
            x_pix_rel = x_pix+shift_to_centre[0]
            y_pix_rel = y_pix+shift_to_centre[1]
            
            x_m = x_pix_rel * self.display_size_m[0] / self.resolution[0]
            y_m = y_pix_rel * self.display_size_m[1] / self.resolution[1]
            
            device = x_pix.device

            gaze_m = (gaze_pix + shift_to_centre) * torch.tensor(self.display_size_m) / torch.tensor(self.resolution)
            gaze_deg = torch.rad2deg(torch.atan( gaze_m/self.distance_m ))
            
            ecc = torch.sqrt( (torch.rad2deg(torch.atan(x_m/self.distance_m))-gaze_deg[0])**2 + (torch.rad2deg(torch.atan(y_m/self.distance_m))-gaze_deg[1])**2 )
        
        return ecc
        
    def get_resolution_magnification( self, eccentricity ):
            # Get the relative magnification of the resolution due to
            # eccentricity.
            # 
            # M = R.get_resolution_magnification(eccentricity)
            # 
            # eccentricity is the viewing angle from the center to the fixation point in degrees.
            
            if not self.fixed_ppd is None:
                M = torch( (1), device=eccentricity.device )
            else:            
                eccentricity = torch.minimum( eccentricity, torch.tensor((89.9)) ) # To avoid singulatities
                
                # pixel size in the centre of the display
                pix_rad = 2*math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m )
                
                delta = pix_rad/2
                tan_delta = math.tan(delta)
                tan_a = torch.tan( torch.deg2rad(eccentricity) )
                
                M = (torch.tan(torch.deg2rad(eccentricity)+delta)-tan_a)/tan_delta

            return M

    def print(self):
        logging.info( 'Geometric display model:' )
        if hasattr( self, "fixed_ppd" ):
            logging.info( '  Fixed pixels-per-degree: {}'.format(self.fixed_ppd) )
        else:
            logging.info( '  Resolution: {w} x {h} pixels'.format( w=self.resolution[0], h=self.resolution[1] ) )
            logging.info( '  Display size: {w:.1f} x {h:.1f} cm'.format( w=self.display_size_m[0]*100, h=self.display_size_m[1]*100) )
            logging.info( '  Display size: {w:.2f} x {h:.2f} deg'.format( w=self.display_size_deg[0], h=self.display_size_deg[1] ) )
            logging.info( '  Viewing distance: {d:.3f} m'.format(d=self.distance_m) )
            logging.info( '  Pixels-per-degree (center): {ppd:.2f}'.format(ppd=self.get_ppd()) )

    @classmethod
    def load( cls, display_name ):
        models_file      = os.path.join(os.path.dirname(__file__), "fvvdp_data/display_models.json")
        models = json2dict(models_file)

        for mk in models:
            if mk == display_name:
                model = models[mk]
                assert "resolution" in model

                inches_to_meters = 0.0254

                W, H = model["resolution"]

                if "fov_diagonal" in model: fov_diagonal = model["fov_diagonal"]
                else:                       fov_diagonal = None

                if   "viewing_distance_meters" in model: distance_m = model["viewing_distance_meters"]
                elif "viewing_distance_inches" in model: distance_m = model["viewing_distance_inches"] * inches_to_meters
                else:                                    distance_m = None

                if   "diagonal_size_meters" in model: diag_size_inch = model["diagonal_size_meters"] / inches_to_meters
                elif "diagonal_size_inches" in model: diag_size_inch = model["diagonal_size_inches"] 
                else:                                 diag_size_inch = None

                obj = fvvdp_display_geometry( (W, H), distance_m=distance_m, fov_diagonal=fov_diagonal, diagonal_size_inches=diag_size_inch)
                return obj

        logging.error("Error: Display model '%s' not found in display_models.json" % display_name)
        return None

