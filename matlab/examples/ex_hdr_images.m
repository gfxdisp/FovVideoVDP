% This example shows how to run FovVideoVDP on HDR images.

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

I_ref = hdrread( 'nancy_church.hdr' );

L_peak = 4000; % Peak luminance of an HDR display

% HDR images are often given in relative photometric units. They MUST be
% mapped to absolute amount of light emitted from the display. For that, 
% we map the peak value in the image to the peak value of the display,
% then we increase the brightness by 2 stops (*4):
I_ref = I_ref/max(I_ref(:)) * L_peak * 4;

% Add Gaussian noise of 20% contrast
I_test_noise = I_ref + I_ref.*randn(size(I_ref))*0.3;

I_test_blur = imgaussfilt( I_ref, 2 );

% We use geometry of SDR 4k 30" display, but ignore its photometric
% properties and instead tell that we pass absolute colorimetric values. 
% Note that many HDR images are in rec709 color space, so no need to
% specify rec2020. 
[Q_JOD_noise, diff_map_noise] = fvvdp( I_test_noise, I_ref, 'display_name', 'standard_hdr', 'display_photometry', 'absolute', 'heatmap', 'threshold' );
[Q_JOD_blur, diff_map_blur] = fvvdp( I_test_blur, I_ref, 'display_name', 'standard_hdr', 'display_photometry', 'absolute', 'heatmap', 'threshold' );

clf
subplot( 1, 2, 1 );
imshow( diff_map_noise );
title( sprintf('Quality: %g JOD\n', Q_JOD_noise ) );

subplot( 1, 2, 2 );
imshow( diff_map_blur );
title( sprintf('Quality: %g JOD\n', Q_JOD_blur ) );
