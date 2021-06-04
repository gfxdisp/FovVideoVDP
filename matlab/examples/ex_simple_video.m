% This example shows how to run FovVideoVD on video and then and visualize its results. 

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

% The frame to use for the video. Note that this is uint16 image
I_ref = imread( 'wavy_facade.png' );
%I_ref = imread( 'night_car.png' );

N = 60; % The number of frames
fps = 30; % Frames per second

V_ref = repmat( I_ref, [1 1 1 N] ); % Reference video (in colour). Use [height x with x N] matrix for a grayscale video. 
max_v = single(intmax( 'uint16' ));
N_amplitude = 0.07; % Amplitude of the noise (in gamma encoded values, scale 0-1)
V_dynamic_noise = V_ref + uint16(randn(size(V_ref))*max_v*N_amplitude); % Dynamic Gaussian noise
V_static_noise = V_ref + repmat( uint16(randn(size(V_ref,1),size(V_ref,2),size(V_ref,3))*max_v*N_amplitude), [1 1 1 size(V_ref,4)] ); % Static Gaussian noise

% Used to compare Python and Matlab versions
% profile='Uncompressed AVI';
% save_as_video( uint8(V_ref/255), 'wavy_facade_vid_ref.avi', 30, profile );
% save_as_video( uint8(V_dynamic_noise/255), 'wavy_facade_vid_dynamic_noise.avi', 30, profile );
% save_as_video( uint8(V_static_noise/255), 'wavy_facade_vid_static_noise.avi', 30, profile );

options = {};
display_name = 'standard_4k';
tic
[Q_JOD_static_noise, diff_map_static_noise] = fvvdp( V_static_noise, V_ref, 'frames_per_second', fps, 'display_name', display_name, 'heatmap', 'threshold', 'options', options );
toc

% 'options', { 'sensitivity_correction', -10, 'csf_sigma', -1.5, 'w_transient', 0.277456 }
tic
[Q_JOD_dynamic_noise, diff_map_dynamic_noise] = fvvdp( V_dynamic_noise, V_ref, 'frames_per_second', fps, 'display_name', display_name, 'heatmap', 'threshold', 'options', options );
toc

fprintf( 1, '=== Static noise Q_JOD = %g\n', Q_JOD_static_noise );
fprintf( 1, '=== Dynamic noise Q_JOD = %g\n', Q_JOD_dynamic_noise );

% Prepare visualization

% Videos are large so better to store them as uint8. 
V_all = cat( 1, cat( 2, uint8(V_static_noise/256), uint8(diff_map_static_noise*256) ), ...
                cat( 2, uint8(V_dynamic_noise/256), uint8(diff_map_dynamic_noise*256) ) );

implay( V_all, fps )

