function ex_paper_teaser()

    % add paths for matlab code
    addpath(genpath('../'));
    % load up the SIGGRAPH logo worldcloud image
    I_ref = im2double( imread( 'SIGGRAPH_wordcloud.png' ) );

    % set the center point of the image
    mid = round([size(I_ref,1) size(I_ref,2)]/2);

    % creating the distorted test images
    I_bl = imgaussfilt( I_ref, 2 );
    I_noise = imnoise( I_ref, 'gaussian', 0, 0.01 );
    % using a .jpg image to generated a compressed test
    fname = strcat( tempname(), '.jpg' );
    imwrite( I_ref, fname, 'Quality', 10 );
    I_jpeg = im2double( imread( fname ) );
    delete( fname );

    % set the eccentricities we will analyze
    ECCs = [0 30 60];
    
    % set up the data structure for the results
    res = cell(length(ECCs),4);
    % define the geometry of the display we will simulate
    disp_geo = fvvdp_display_geometry( [3840 2160], 'diagonal_size_inches', 30, 'distance_m', 0.6 );
    % define the framerate of the display we will simulate
    fps = 60;
    
    % for each eccentricity
    for ee=1:length(ECCs)
        % and for each test case
        for dd=1:4

            switch dd
                case 1
                    I_test = I_bl;
                case 2
                    I_test = I_jpeg;
                case 3
                    I_test = I_ref;
                case 4
                    I_test = I_noise;
            end
            
            % we generate the luminance images using a simple gamma model
            gamma = 2.2;
            L_max = 100;
            L_ref = L_max*get_luminance_image(I_ref).^gamma;
            L_test = L_max*get_luminance_image(I_test).^gamma;

            % add flicker for the flicker test
            if 1
                N = 30;
                [xx, yy] = meshgrid( 1:size(L_ref,2), 1:size(L_ref,1) );
                x0 = size(L_ref,2)*0.25;
                y0 = size(L_ref,1)*0.75;
                R = sqrt( (xx-x0).^2 + (yy-y0).^2 );
                mask = R<20;
                L_fl = L_test;
                dim_factor = 0.8;
                L_fl(mask) = L_test(mask)*dim_factor;
                L_ref = repmat( L_ref, [1 1 N] );
                L_test = repmat( L_test, [1 1 N] );
                for kk=1:2:N
                    L_test(:,:,kk) = L_fl;
                end
            end
            % set the fixation point based on the eccentricity
            fixation_point = [mid(2) - 0.25.*ECCs(ee)*disp_geo.get_ppd()  mid(1)];
            % run the metric with all the parameters set above 
            [res{ee,dd}.Q_JOD, res{ee,dd}.diff_map] = fvvdp( L_test, L_ref, 'display_photometry', 'absolute', 'frames_per_second', fps, ...
                'display_name', 'standard_hmd', 'foveated', true, 'quiet', true, 'options', {'fixation_point', fixation_point, 'use_gpu', false} );
        end
    end

    %%
    clf;
    % stich up an image with the reference and distorted tests
    I_test_vis = cat( 1, cat( 2, I_bl(1:mid(1),1:mid(2),:), I_jpeg(1:mid(1),(mid(2)+1):end,:) ),...
        cat( 2, I_ref((mid+1):end,1:mid(2),:), I_noise((mid(1)+1):end,(mid(2)+1):end,:) ) );
    % accurately de-gamma the flicker test
    mask3 = repmat(mask,[1 1 3]);
    I_test_vis(mask3) = I_test_vis(mask3)*dim_factor.^(1/gamma);
    I_vis = cat( 2, I_ref, I_test_vis );
    JODs = zeros(4,3);


    for ee=1:length(ECCs)

        df_new = cat( 1, cat( 2, res{ee,1}.diff_map(1:mid(1),1:mid(2),end), res{ee,2}.diff_map(1:mid(1),(mid(2)+1):end,end) ),...
        cat( 2, res{ee,3}.diff_map((mid+1):end,1:mid(2),end), res{ee,4}.diff_map((mid(1)+1):end,(mid(2)+1):end,end) ) );


        I_vis = cat( 2, I_vis, hdrvdp_visualize( 'pmap', df_new, { 'context_image', get_luminance_image(I_ref) } ) );

        fprintf( 1, '=== ecc = %g\n', ECCs(ee) );
        for dd=1:4
            fprintf( 1, '%g\n', res{ee,dd}.Q_JOD );
            JODs(dd,ee) = res{ee,dd}.Q_JOD;
        end        

    end
    
%     figure(2)
%     clf
%     for dd=1:4
%         plot( ECCs, JODs(dd,:) );
%         hold on
%     end
%     legend( 'Blur', 'JPEG', 'Flicker', 'Noise' );


    figure(1)
    imshow( I_vis );
    imwrite( I_vis, 'teaser_background.png' );

    disp('done!');

end

function Y = get_luminance_image(img)
    Y = img(:,:,1) * 0.212656 + img(:,:,2) * 0.715158 + img(:,:,3) * 0.072186;
end