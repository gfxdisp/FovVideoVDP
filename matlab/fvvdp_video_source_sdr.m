classdef fvvdp_video_source_sdr < fvvdp_video_source
% A video source that takes loaded video frames as (height x width x 3 x
% frames) matrices. This class is for display-encoded (gamma-encoded)
% content that must be processed by a display model to produce linear
% absolute luminance emitted from a display.
    
    properties
        fps;
        test_video;
        reference_video;        
        is_video;
        is_color;                
        display_model;
        color_to_luminance;
    end    
    
    methods
        
        function vs = fvvdp_video_source_sdr( test_video, reference_video, fps, display_model, color_space_name )
            assert( all( size(test_video)==size(reference_video) ) );

            if any( size(test_video)~=size(reference_video) )
                error( 'Test and reference image/video tensors must be exactly the same size' );
            end
            
            if fps==0 && ( (ndims(test_video)==3 && size(test_video,3)>3) || (ndims(test_video)==4 && size(test_video,4)>1) )
                error( 'When passing video sequences, you must set ''frames_per_second'' parameter' );
            end
            
            vs.fps = fps;
            vs.is_video = (fps>0);            
            vs.is_color = (~vs.is_video && size(test_video,3)==3) || (vs.is_video && length(size(test_video))==4 && size(test_video,3)==3);
            vs.test_video = test_video;
            vs.reference_video = reference_video;
            
            col_spaces = vs.load_json( 'color_spaces.json' );
            
            if ~exist( 'color_space', 'var' ) || isempty( color_space_name )
                color_space_name = 'sRGB';
            end
            
            if ~isfield( col_spaces, color_space_name )
                error( 'Unknown color space "%s". Check display_models/color_spaces.json for available color spaces.', color_space_name );
            end
            cs = col_spaces.(color_space_name);
            vs.color_to_luminance = cs.RGB2Y;                                                           
            
            if ~exist( 'display_model', 'var' ) || isempty( display_model )
                display_model = 'sdr_4k_30';
            end
            
            if ischar( display_model )
                disp_models = vs.load_json( 'display_models.json' );

                if ~isfield( disp_models, display_model )
                    error( 'Unknown display model "%s". Check display_models/display_models.json for available display models.', display_model );
                end
                dm = disp_models.(display_model);
                            
                vs.display_model = fvvdp_display_photometry.create_from_json( dm );
            else
                vs.display_model = display_model;
            end
            
        end

        function fps = get_frames_per_second(vs)
            fps = vs.fps;
        end
        
        % Return a [height width frames] vector with the resolution and
        % the number of frames in the video clip. [height width 1] is
        % returned for an image. 
        function sz = get_video_size(vs)
            if vs.is_color
                sz = [size(vs.reference_video,1) size(vs.reference_video,2) size(vs.reference_video,4)];
            else
                sz = size(vs.reference_video);
                if length(sz)==2
                    sz = [ sz 1 ];
                end
            end
        end
        
        % Get a test video frame as a single-precision luminance map
        % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
        % starting from 1. If use_gpu==true, the function should return a
        % gpuArray.
        function L_test = get_test_frame( vs, frame, use_gpu )
            L_test = vs.get_frame_(vs.test_video, frame, use_gpu);            
        end
        
        function L_ref = get_reference_frame( vs, frame, use_gpu )
            L_ref = vs.get_frame_(vs.reference_video, frame, use_gpu);            
        end
        
        function L = get_frame_( vs, from_video, frame, use_gpu )
           
            % Determine the maximum value of the data type storing the
            % image/video
            if isa( from_video, 'uint8' ) || isa( from_video, 'uint16' )
                peak_value = single(intmax( class(from_video) ));
            elseif isa( from_video, 'logical' ) || isa( from_video, 'single' ) || isa( from_video, 'double' )
                peak_value = 1.0;
            else
                error( 'unknown data type of the video/image' );
            end
            
            if vs.is_color
                L = single(from_video(:,:,:,frame));
            else
                L = single(from_video(:,:,frame));
            end
            
            if use_gpu
                L = gpuArray(L);
            end
            L = vs.display_model.forward( L/peak_value );
            
            if vs.is_color
                % Convert to grayscale
                L = L(:,:,1)*vs.color_to_luminance(1) + L(:,:,2)*vs.color_to_luminance(2) + L(:,:,3)*vs.color_to_luminance(3);
            end
            
        end

        
    end

    
    
end