% This example shows how to run FovVideoVDP on video, assuming foveated
% viewing.
%
% This example simulates viewing in a VR headset. Therefore, the noise is
% much less visible on a monitor that it is in a VR headset, which has much
% smaller effective resolution (in terms of pixels per degree).

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

% The frame to use for the video. Note that this is uint16 image
I_ref = imread( 'wavy_facade.png' );
ar = 1440/1600; % the aspect ratio of HTC Vive Pro (width/height)
crop_pix = floor((size(I_ref,2) - size(I_ref,1)*ar)/2);
I_ref = I_ref(:,crop_pix:(end-crop_pix),:); % Crop to the aspect ratio of HTC Vive Pro
I_ref = imresize( I_ref, [1600 1440], 'bicubic' ); % Upscale the image to match the resolution of HTC Vive Pro

N = 60; % The number of frames
fps = 30; % Frames per second

V_ref = repmat( I_ref, [1 1 1 N] ); % Reference video (in colour). Use [height x with x N] matrix for a grayscale video. 
max_v = single(intmax( 'uint16' ));
noise_ampitude = 0.02;
V_dynamic_noise = V_ref + uint16(randn(size(V_ref))*max_v*noise_ampitude); % Dynamic Gaussian noise

% The gaze will move from the top-left to the bottom-right corner
% We are pasing [N 2] matrix with the fixation points as [x y], where x
% goes from 0 to width-1.
% If the gaze position is fixed, pass [x y] vector. 
% If you ommit 'fixation_point' option, the fixation will be set to the
% centre of the image.
gaze_pos = [linspace(0,size(V_ref,2)-1,N)' linspace(0,size(V_ref,1)-1,N)'];
options = { 'fixation_point', gaze_pos };

tic
[Q_JOD_dynamic_noise, diff_map_dynamic_noise] = fvvdp( V_dynamic_noise, V_ref, 'frames_per_second', fps, 'display_name', 'htc_vive_pro', 'heatmap', 'threshold', 'foveated', true, 'options', options );
toc

fprintf( 1, '=== Dynamic noise Q_JOD = %g\n', Q_JOD_dynamic_noise );

% Prepare visualization

% Videos are large so better to store them as uint8. 
% Add fixation points
V_dynamic_noise_fp = fvvdp_add_fixation_cross( uint8(V_dynamic_noise/256), gaze_pos );
V_diff_map_fp = fvvdp_add_fixation_cross( uint8(diff_map_dynamic_noise*256), gaze_pos );
V_all = cat( 1, cat( 2, V_dynamic_noise_fp, V_diff_map_fp ) );

implay( V_all, fps )

