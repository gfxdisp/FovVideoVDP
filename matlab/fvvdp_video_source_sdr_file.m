classdef fvvdp_video_source_sdr_file < fvvdp_video_source
% A video source that loads frames from video files using Matlab's
% VideoReader. 
    
    properties
        fps;
        test_video;
        reference_video;        
        is_video;
        is_color;                
        display_model;
        color_to_luminance;
        
        test_reader;
        reference_reader;
    end    
    
    methods
        
        function vs = fvvdp_video_source_sdr_file( test_video, reference_video, display_model, color_space_name )
            
            vs.test_reader = VideoReader( test_video );
            vs.reference_reader = VideoReader( reference_video );
            
            if abs(vs.reference_reader.FrameRate-vs.test_reader.FrameRate)>1e-3
                error( 'Test and reference video must have the same frame rate. Found %g and %g fps.', vs.test_reader.FrameRate, vs.reference_reader.FrameRate );
            end
            
            vs.fps = vs.reference_reader.FrameRate;
            vs.is_video = true;            
            vs.is_color = true; % Not sure how to handle BW videos (~vs.is_video && size(test_video,3)==3) || (vs.is_video && length(size(test_video))==4 && size(test_video,3)==3);
            
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
            sz = [vs.reference_reader.Height vs.reference_reader.Width vs.reference_reader.NumFrames];
        end
        
        % Get a test video frame as a single-precision luminance map
        % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
        % starting from 1. If use_gpu==true, the function should return a
        % gpuArray.
        function L_test = get_test_frame( vs, frame, use_gpu )
            L_test = vs.get_frame_(vs.test_reader, frame, use_gpu);            
        end
        
        function L_ref = get_reference_frame( vs, frame, use_gpu )
            L_ref = vs.get_frame_(vs.reference_reader, frame, use_gpu);            
        end
        
        function L = get_frame_( vs, vid_reader, frame, use_gpu )
           
            frame = read(vid_reader,frame);
            
            % Determine the maximum value of the data type storing the
            % image/video
            if isa( frame, 'uint8' ) || isa( frame, 'uint16' )
                peak_value = single(intmax( class(frame) ));
            elseif isa( frame, 'logical' ) || isa( frame, 'single' ) || isa( frame, 'double' )
                peak_value = 1.0;
            else
                error( 'unknown data type of the video/image' );
            end
            
            L = single(frame);            
            
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