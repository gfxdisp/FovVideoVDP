% This example shows how to call FovVideoVDP while specifying the display
% photometric data and, in particular, display brightness. The example plots image 
% quality assessed on displays of different brightness. It demonstrates that 
% image distortions become less visible as the display becomes darker. 

if ~exist( 'fovvdp', 'file' )
    addpath( fullfile( pwd, '..') );
end

I_ref = imread( 'wavy_facade.png' );
I_test_noise = imnoise( I_ref, 'gaussian', 0, 0.001 );

% Measure quality on displays of different brightness
disp_peaks = logspace( log10(1), log10(1000), 5 );

% Display parameters
contrast = 1000;  % Display contrast 1000:1
gamma = 2.2;      % Standard gamma-encoding
E_ambient = 100;  % Ambient light = 100 lux
k_refl = 0.005;   % Reflectivity of the display

Q_JOD = zeros(length(disp_peaks),1);
for dd=1:length(disp_peaks)

    Y_peak = disp_peaks(dd);     % Peak luminance in cd/m^2 (the same as nit)
    disp_photo = fvvdp_display_photo_gog( Y_peak, contrast, gamma, E_ambient, k_refl );
    
    Q_JOD(dd) = fvvdp( I_test_noise, I_ref, 'frames_per_second', 0, 'display_photometry', disp_photo );
end

clf
plot( disp_peaks, Q_JOD, '-o' );
grid on;
set( gca, 'XScale', 'log' );
xlabel( 'Display peak luminance [cd/m^2]' );
ylabel( 'Quality [JOD]' );


