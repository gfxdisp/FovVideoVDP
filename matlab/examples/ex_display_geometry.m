% This example shows how to call FovVideoVDP while specifying the display
% size and viewing distance. The example plots image quality assessed at
% several viewing distances. It demonstrates that image distortions become
% less visible as the viewing distance is increased. 

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

I_ref = imread( 'wavy_facade.png' );
I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.005 );


% Measure quality at several viewing distances
distances = linspace( 0.5, 2, 5 );

Q_JOD = zeros(length(distances),1);
for dd=1:length(distances)
    % 4K, 30 inch display, seen at different viewing distances
    disp_geo = fvvdp_display_geometry( [3840 2160], 'diagonal_size_inches', 30, 'distance_m', distances(dd) );
    Q_JOD(dd) = fvvdp( I_test_noise, I_ref, 'frames_per_second', 0, 'display_geometry', disp_geo );
end

clf
plot( distances, Q_JOD, '-o' );
grid on;
xlabel( 'Viewing distance [m]' );
ylabel( 'Quality [JOD]' );


