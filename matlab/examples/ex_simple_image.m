% This example shows how to call FovVideoVDP and visualize its results. 

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

I_ref = imread( '../../example_media/wavy_facade.png' );
%I_ref = imread( 'tree.jpg' );

% Generate distorted images as needed.
% We store distorted images so that we can compare Matlab/Python results
noise_fname = '../../example_media/wavy_facade_noise.png';
if isfile( noise_fname )
    I_test_noise = imread( noise_fname );
else
    I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.003 );
    imwrite( I_test_noise, noise_fname );
end

blur_fname = '../../example_media/wavy_facade_blur.png';
if isfile( blur_fname )
    I_test_blur = imread( blur_fname );
else
    I_test_blur = imgaussfilt( I_ref, 2 );
    imwrite( I_test_blur, blur_fname );
end

[Q_JOD_noise, diff_map_noise] = fvvdp( I_test_noise, I_ref, 'display_name', 'standard_4k', 'heatmap', 'threshold' );
[Q_JOD_blur, diff_map_blur] = fvvdp( I_test_blur, I_ref, 'display_name', 'standard_4k', 'heatmap', 'threshold' );

fprintf( 1, '=== Noise Q_JOD = %g\n', Q_JOD_noise);
fprintf( 1, '=== Blur Q_JOD = %g\n', Q_JOD_blur);


clf
subplot( 2, 2, 1 );
imshow( I_test_noise );
title( 'Test image with noise' );

subplot( 2, 2, 3 );
imshow( diff_map_noise );
title( sprintf('Noise; Quality: %.3f JOD\n', Q_JOD_noise ) );


subplot( 2, 2, 2 );
imshow( I_test_blur );
title( 'Test image with blur' );

subplot( 2, 2, 4 );
imshow( diff_map_blur );
title( sprintf('Blur; Quality: %.3f JOD\n', Q_JOD_blur ) );
