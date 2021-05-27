classdef fvvdp_display_photometry
    % An abstract class for the classes that describes display photometry
    % and can convert between display-encoded and linear color values
       
    methods( Abstract )
        
        % Forward display model. 
        % Transforms gamma-correctec pixel values V, which must be in the range
        % 0-1, into absolute linear colorimetric values emitted from
        % the display.
        L = forward( dm, V );
        
        % The the peak luminance of the display in cd/m^2
        Y_peak = get_peak_luminance( dm );
        
        % The effective black level of the display (with the screen
        % reflections) in cd/m^2
        Y_black = get_black_level( dm );
        
        % Display information about the simulated display. 
        print( dm );
        
    end
    
    methods( Static )
        
        function dm = create_from_json( dm_struct )
            
            if isfield( dm_struct, 'contrast' )
                contrast = dm_struct.contrast;
            else
                contrast = dm_struct.max_luminance/dm_struct.min_luminance;
            end
            
            if isfield( dm_struct, 'E_ambient' )
                E_ambient = dm_struct.E_ambient;
            else
                E_ambient = 0;
            end            

            if isfield( dm_struct, 'k_refl' )
                k_refl = dm_struct.k_refl;
            else
                k_refl = [];
            end
            
            dm = fvvdp_display_photo_gog( dm_struct.max_luminance, contrast, [], E_ambient, k_refl );            
            
        end
        
    end
    
end