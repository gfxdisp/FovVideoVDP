classdef fvvdp_display_photo_absolute < fvvdp_display_photometry
    % Use this photometric model when passing absolute colorimetric of
    % photometric values, scaled in cd/m^2
    
    properties
        L_min = 0.0001;
    end
    
    methods
                
        function L = forward( dm, V )
            % Forward display model. 
            % Clamps the valies that are below L_min.
            
            L = max( dm.L_min, V );
            
            if max(V(:))<1
                warning( 'fovvdp:low_intensities', 'Pixel valies are very low. Perhaps images are not scaled in absolute units of cd/m^2.');
            end
                        
        end
        
        function print( dm )
            fprintf( 1, 'Photometric display model:\n' );
            fprintf( 1, '  Absolute photometric/colorimetric values\n' );
        end
        
    end    
    
end