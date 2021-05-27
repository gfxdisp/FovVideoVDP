classdef fvvdp_display_photo_absolute < fvvdp_display_photometry
    % Use this photometric model when passing absolute colorimetric of
    % photometric values, scaled in cd/m^2
    
    properties
        L_max = 10000;
        L_min = 0.005;
    end
    
    methods
                
        function dm = fvvdp_display_photo_absolute( L_max, L_min )
            
            if exist( 'L_max', 'var' )
                dm.L_max = L_max;
            end

            if exist( 'L_min', 'var' )
                dm.L_min = L_min;
            end
            
        end
        
        function L = forward( dm, V )
            % Forward display model. 
            
            % Clamp the valies that are outside the (L_min, L_max) range.           
            L = min(max( dm.L_min, V ), dm.L_max);
            
            if max(V(:))<1
                warning( 'fovvdp:low_intensities', 'Pixel valies are very low. Perhaps images are not scaled in the absolute units of cd/m^2.');
            end
                        
        end
        
        function Y_peak = get_peak_luminance( dm )
            Y_peak = dm.L_max;
        end

        function Y_black = get_black_level( dm )
            Y_black = dm.L_min;
        end
        
        function print( dm )
            fprintf( 1, 'Photometric display model:\n' );
            fprintf( 1, '  Absolute photometric/colorimetric values\n' );
        end
        
    end    
    
end