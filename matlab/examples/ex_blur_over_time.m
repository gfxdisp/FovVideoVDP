% This example shows how to run FovVideoVD on video and then and visualize its results. 
% In this example, we modulate the amount of blur over time. 

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

% The frame to use for the video. Note that this is a uint16 image
I_ref = imread( 'tree.jpg' );

N = 60*4; % The number of frames
fps = 30; % Frames per second

V_ref = repmat( I_ref, [1 1 1 N] ); % Reference video (in colour). Use [height x with x N] matrix for a grayscale video. 
max_v = single(intmax( 'uint16' ));

V_blur = zeros(size(V_ref), 'like', V_ref);
sigma_max = 2;
SIGMAs = cat( 2, linspace( 0.01, sigma_max, N/2 ), linspace( sigma_max, 0.01, N/2 ) );
for kk=1:N % For each frame
    sigma = SIGMAs(kk);
    V_blur(:,:,:,kk) = imgaussfilt(V_ref(:,:,:,kk), sigma);
end

options = {};
display_name = 'standard_4k';
tic
[Q_JOD_blur, diff_map_blur] = fvvdp( V_blur, V_ref, 'frames_per_second', fps, 'display_name', display_name, 'heatmap', 'threshold', 'options', options );
toc

% Prepare visualization

% Videos are large so better to store them as uint8. 
V_all = cat( 2, uint8(V_blur), uint8(diff_map_blur*256) );

implay( V_all, fps )

