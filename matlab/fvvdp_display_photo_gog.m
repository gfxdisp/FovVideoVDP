classdef fvvdp_display_photo_gog < fvvdp_display_photometry
    % Gain-gamma-offset display model to simulate SDR displays
    
    properties
        Y_peak;
        contrast;
        gamma;
        E_ambient;
        k_refl;
    end
    
    methods
        
        function dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient, k_refl )
            % Gain-gamma-offset display model to simulate SDR displays
            %
            % dm = fvvdp_display_photo_gog( Y_peak )
            % dm = fvvdp_display_photo_gog( Y_peak, contrast )
            % dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma )
            % dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient )
            % dm = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient, k_refl )
            %
            % Parameters (default value shown in []):
            % Y_peak - display peak luminance in cd/m^2 (nit), e.g. 200 for a typical
            %          office monitor
            % contrast - [1000] the contrast of the display. The value 1000 means
            %          1000:1
            % gamma - [-1] gamma of the display, typically 2.2. If -1 is
            %         passed, sRGB non-linearity is used.         
            % E_ambient - [0] ambient light illuminance in lux, e.g. 600 for bright
            %         office
            % k_refl - [0.005] reflectivity of the display screen
            %
            % For more details on the GOG display model, see:
            % https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016perceptual_display.pdf
            %
            % Copyright (c) 2010-2021, Rafal Mantiuk
            
            dm.Y_peak = Y_peak;
            
            if ~exist( 'contrast', 'var' ) || isempty( contrast )
                dm.contrast = 1000;
            else
                dm.contrast = contrast;
            end
            
            if ~exist( 'gamma', 'var' ) || isempty( gamma )
                dm.gamma = -1;
            else
                dm.gamma = gamma;
            end
            
            if ~exist( 'E_ambient', 'var' ) || isempty( E_ambient )
                dm.E_ambient = 0;
            else
                dm.E_ambient = E_ambient;
            end
            
            if ~exist( 'k_refl', 'var' ) || isempty( k_refl )
                dm.k_refl = 0.005;
            else
                dm.k_refl = k_refl;
            end
            
        end
        
        function L = forward( dm, V )
            % Transforms gamma-correctec pixel values V, which must be in the range
            % 0-1, into absolute linear colorimetric values emitted from
            % the display.
            
            if any(V(:)>1) || any(V(:)<0)
                warning( 'fovvdp:pixels_out_of_range', 'Pixel values must be in the range 0-1');
            end
            
            Y_black = dm.get_black_level();
            
            if dm.gamma==-1
                L = (dm.Y_peak-Y_black)*srgb2lin(V) + Y_black;
            else
                L = (dm.Y_peak-Y_black)*(V.^dm.gamma) + Y_black;
            end
            
        end
        
        function Y_peak = get_peak_luminance( dm )
            Y_peak = dm.Y_peak;
        end
        
        function Y_black = get_black_level( dm )
            Y_refl = dm.E_ambient/pi*dm.k_refl; % Reflected ambient light            
            Y_black = Y_refl + dm.Y_peak/dm.contrast;
        end
        
        function print( dm )
            Y_black = dm.get_black_level();
            
            fprintf( 1, 'Photometric display model:\n' );
            fprintf( 1, '  Peak luminance: %g cd/m^2\n', dm.Y_peak );
            fprintf( 1, '  Contrast - theoretical: %d:1\n', round(dm.contrast) );
            fprintf( 1, '  Contrast - effective: %d:1\n', round(dm.Y_peak/Y_black) );
            fprintf( 1, '  Ambient light: %g lux\n', dm.E_ambient );
            fprintf( 1, '  Display reflectivity: %g%%\n', dm.k_refl*100 );            
        end
        
    end    
    
    
end