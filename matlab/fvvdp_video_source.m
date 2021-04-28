classdef fvvdp_video_source
   
    methods( Abstract )
   
        % Return [height x width x frames] vectors with the resolution and
        % the length of the video clip.
        sz = get_video_size(vs);

        % Return the frame rate of the video
        fps = get_frames_per_second(vs);
        
        % Get a pair of test and reference video frames as a single-precision luminance map
        % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
        % starting from 1. If use_gpu==true, the function should return a
        % gpuArray.
        L_test = get_test_frame( vs, frame, use_gpu );                
        L_ref = get_reference_frame( vs, frame, use_gpu );        
        
        
    end

    methods
        function L_vid = get_reference_video( vs, use_gpu )
            video_sz = vs.get_video_size();
            if use_gpu
                L_vid = zeros( video_sz, 'gpuArray', 'single' );
            else
                L_vid = zeros( video_sz, 'single' );
            end
            
            for ff=1:video_sz(3)
                L_vid(:,:,ff) = vs.get_reference_frame(ff,use_gpu);
            end
        end
    end
    
    
    methods( Static )

        function val = load_json( fname )
            fname_full = fullfile( fvvdp_data_dir(), fname );            
            if ~isfile( fname_full )
                error( 'Cannot find the configuration file "%s"', fname_full );
            end
            fid = fopen(fname_full);
            raw = fread(fid,inf);
            str = char(raw');
            fclose(fid);
            val = jsondecode(str);
        end
        
    end
    
end
