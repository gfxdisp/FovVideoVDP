function res = fvvdp_core( video_source, display_geometry, options )
% Foveated-Video-Visible-Differce-Predictor or FovVideoVDP
% This is a low-level interface to the metric. If unsure, use fvvdp
% instead.
%
% res = fvvdp_core( video_source, display_geometry, options )
%
% video_source - object of the class fvvdp_video_source
% display_geometry - object of the class fvvdp_display_geometry
% options - a cell array with { 'name', value } pairs. Refer to the
%           documentation of fvvdp() for the most important options, or
%           checke the comments next to the masking_par in this file for
%           the description of all the options. 
%
% 

frames_per_s  = video_source.get_frames_per_second();

if ~exist( 'options', 'var' )
    options = {};
end

pix_per_deg = display_geometry.get_ppd();

% Check the path and add the missing directories as needed
if ~exist( 'get_temporal_filters', 'file' )
    pathstr = fileparts(mfilename( 'fullpath' ));
    addpath( fullfile( pathstr, 'utils' ) );
end


% Calibration parameters are read from JSON file to make sure they are
% consistent between Matlab and Python versions of the metric.
metric_par = fvvdp_load_parameters_from_json();

% The calibration parameters (normally loaded from the JSON file)
% metric_par.mask_p = 2.4;  % The exponent of the exitation signal of the
                           % masking model
% metric_par.mask_c = -0.544583; % log10() of the coefficient of the masking
                           % signal; "k" in the paper
% metric_par.pu_dilate = 0; % Spatial integration of the inhibitory masking
                           % signal - not used in the main model
% metric_par.w_transient = 0.25; % The weight of the transient temporal channel
% metric_par.beta = 0.95753; % The exponent of the spatial summation (p-norm)
% metric_par.beta_t = 1; % The exponent of the summation over time (p-norm)
% metric_par.beta_tch = 0.684842; % The exponent of the summation over temporal channels (p-norm)
% metric_par.beta_sch = 1; % The exponent of the summation over spatial channels (p-norm)
% metric_par.filter_len = -1; % The length of the temporal filter. Set to -1 for an adaptive length
% metric_par.sustained_sigma = 0.5; % Parameter of the sustained channel temporal filter. Eq. 3 in the paper
% metric_par.sustained_beta = 0.06; % Parameter of the sustained channel temporal filter. Eq. 3 in the paper
% metric_par.csf_sigma = -1.5;  % The size of the stimulus passed to the CSF. Wavelength if negative. \sigma_0 in the paper (if negative).
% metric_par.sensitivity_correction = 10; % Sensitity adjustmemnt of the CSF in dB. Negative values make the metric less sensitive. Note that this is 20*log10(S_corr) from the paper. 
% metric_par.masking_model = 'min_mutual_masking_perc_norm2'; % The ID of the masking model
% metric_par.local_adapt = 'gpyr';  % Local adaptation: 'simple' or or 'gpyr'
% metric_par.contrast = 'weber';  % Either 'weber' or 'log'
% metric_par.jod_a = -0.249449; % The alpha parameter of the JOD regression. Eq. 19 in the paper. 
% metric_par.log_jod_exp = log10(0.372455); % The log10() of the beta parameter of the JOD regression. Eq. 19 in the paper. 
% metric_par.mask_q_sust = 3.23726;  % The exponent of the inhibitory signal in the masking model, sustained channel. q_S in the paper.
% metric_par.mask_q_trans = 3.02625; % The exponent of the inhibitory signal in the masking model, trasient channel. q_T in the paper.
% metric_par.k_cm = 0.405835;  % The parameter controlling the effect of cortical magnification. See eq. 11 in the paper.

% Operational options/parameters
metric_par.do_foveated = false;
metric_par.frame_padding = 'replicate'; % How to pad frame at the beginning of the video (for the temporal filters): 'replicate' or 'circular'
metric_par.use_gpu = true; % Set to false to disable processing on a GPU (eg. when CUDA is not supported)
metric_par.do_diff_map = false; % Produce a map of the differences between test and reference images
metric_par.debug = false;       % [internal]: Enable debugging
metric_par.fixation_point = []; % in pixel coordinates (x,y), where x=0..width-1 and y=0..height-1
                                % fixation_point can be also an [N 2] matrix
                                % where N is the number of frames. 
metric_par.band_callback = [];  % [internal] Used to analyze the masking model
metric_par.video_name = 'channels'; % Where to store "debug" video
metric_par.do_temporal_channels = true;  % [internal] Set to false to disable temporal channels and treat each frame as a image (for an ablation study)



metric_par.content_mapping = [];

for kk=1:2:length(options)
    if ~isfield( metric_par, options{kk} )
        error( 'Unknown option "%s"', options{kk} );
    end
    metric_par.(options{kk}) = options{kk+1};
end

if metric_par.use_gpu && gpuDeviceCount==0
    metric_par.use_gpu = false;
    warning( 'No CUDA detected. The metric will use CPU but the computation will be much slower. Pass the option { ''use_gpu'', false } to disable this warning message.' );
end

if metric_par.filter_len == -1
    % For the best accuracy the filter should cover 250ms
    metric_par.filter_len = ceil( 250 / (1000/frames_per_s) );
end

video_sz = video_source.get_video_size();
if isempty( metric_par.fixation_point )
    % The default fixation point is the centre of the frame
    metric_par.fixation_point = round( [video_sz(2) video_sz(1)]/2-1 );
end

if size(metric_par.fixation_point,2) ~= 2 || (size(metric_par.fixation_point,1)~=1 && size(metric_par.fixation_point,1)~=video_sz(3))
    error( '''fixation_point'' must be a [1 2] or [N 2] matrix.' );
end

% if ~isempty( metric_par.log_jod_a )
%     metric_par.jod_a = -10^metric_par.log_jod_a;
% end

% ===== Split into sustained and transient channels

is_image = numel(video_sz)==2 || (video_sz(3) == 1) || ~metric_par.do_temporal_channels; % Are we testing an image or video

if is_image
    temp_ch = 1; % How many temporal channels
    omega = 0; % 0 Hz frequency
else
    temp_ch = 2;
    [F, omega] = get_temporal_filters( frames_per_s, metric_par.filter_len, ...
        metric_par.sustained_sigma, metric_par.sustained_beta );
    F{1} = reshape( single(F{1}), [1 1 numel(F{1})] );
    F{2} = reshape( single(F{2}), [1 1 numel(F{2})] );
    
    if metric_par.use_gpu
        F{1} = gpuArray( F{1} );
        F{2} = gpuArray( F{2} );
    end
end

ms = hdrvdp_lpyr_dec();
if strcmp( metric_par.local_adapt, 'gpyr' )
    % Only this adaptation model can reuse expanded levels of the Gaussian
    % pyramid to reduce the computation.
    ms.do_gauss_exp = true;
else
    ms.do_gauss = true;
end

if numel(video_sz)==2
    N = 1;
else
    N = video_sz(3); % How many frames
end

if( metric_par.debug ) % For debugging
    D_debug = cell(2,1);
    D_debug{1} = zeros(video_sz(1:2)/2);
    D_debug{2} = zeros(video_sz(1:2)/2);
end

if metric_par.do_diff_map
    % Map of the differences between test and reference images
    Dmap = zeros( video_sz );
end

csf = CSF_st_fov();
csf.use_gpu = metric_par.use_gpu;

N_nCSF = [];

sw_buf = cell(2,1); % A sliding window buffer (sustained, transient)

Q_per_ch = [];

% Weights of the sustained and transient channels
w_temp_ch = [1 metric_par.w_transient];

for ff=1:N % for each frame
    
    if ~is_image
        
        fl = metric_par.filter_len;
        
        if ff==1 % First frame
            % Fill the sliding window buffer with the first frame
            switch metric_par.frame_padding
                case 'replicate'
                    sw_buf{1} = repmat( video_source.get_test_frame(1, metric_par.use_gpu), [1 1 fl] );
                    sw_buf{2} = repmat( video_source.get_reference_frame(1, metric_par.use_gpu), [1 1 fl] );
                case 'circular'
                    for kk=1:fl
                        vf = mod(N-1-fl+1+kk-1+1, N)+1; % which frame to copy
                        sw_buf{1}(:,:,kk) = video_source.get_test_frame(vf, metric_par.use_gpu);
                        sw_buf{2}(:,:,kk) = video_source.get_reference_frame(vf, metric_par.use_gpu);
                    end
                otherwise
                    error( 'Unknown frame_padding mode "%s"', metric_par.frame_padding );
            end
        else
            % Slide and add a new frame
            sw_buf{1}(:,:,1:(end-1)) = sw_buf{1}(:,:,2:end);
            sw_buf{2}(:,:,1:(end-1)) = sw_buf{2}(:,:,2:end);
            sw_buf{1}(:,:,end) = video_source.get_test_frame(ff, metric_par.use_gpu);
            sw_buf{2}(:,:,end) = video_source.get_reference_frame(ff, metric_par.use_gpu);
        end
    end
    
    if is_image
        R = cat( 3, video_source.get_test_frame(1, metric_par.use_gpu), ...
            video_source.get_reference_frame(1, metric_par.use_gpu) );
    else
        % For faster processing pack (T_sustained, R_sustained,
        % T_transient, R_transient) into a tensor
        R = zeros(video_sz(1), video_sz(2), 4, 'like', sw_buf{1});
        for cc=1:temp_ch % for each temp channel [sustained, transient]
            for rr=1:2 % for test and reference
                % Filter with the temporal channel filter (sustained or transient)
                R(:,:,(cc-1)*2+rr) = convn( sw_buf{rr}, F{cc}, 'valid' );
            end
        end
    end
    B = ms.decompose( R, pix_per_deg );
    
    switch metric_par.local_adapt
        case 'simple'
            % We use 0.5 deg Gaussian filter, which is a rough approximate of local
            % adaptation
            L_adapt = R(:,:,2); % Use reference sustained
            if strcmp( metric_par.contrast, 'log' )
                L_adapt = 10.^L_adapt;
            end
            L_adapt = imgaussfilt( L_adapt, 0.5*pix_per_deg );
        
        case 'global'
            % Use reference transient channel for the global adaptation
            L_adapt = geomean( reshape( R(:,:,2), [video_sz(1)*video_sz(2) 1] )  );
    end
    
    if isempty( N_nCSF ) || size(metric_par.fixation_point,1)>1 % moving fixation point
        N_nCSF = cell( (B.band_count()-1), 2 ); % Neural noise - the inverse of the CSF
    end
    
    if metric_par.do_diff_map
        Dmap_pyr = ms.create( [video_sz(1) video_sz(2)], pix_per_deg );
    end
    
    L_bkg_bb = [];
    for cc=1:temp_ch % for each temp channel [sustained, transient]
        
        rho_band = B.get_freqs();
        
        if( metric_par.debug ) % For debugging
            D_dd = ms.decompose( zeros(video_sz(1),video_sz(2)), pix_per_deg );
        end
                
        for bb=1:(B.band_count()-1) % for each band except the base band
            
            bnd = B.get_band(bb,1);
            T_f = bnd(:,:,1+(cc-1)*2);
            R_f = bnd(:,:,2+(cc-1)*2);
            
            if isempty( L_bkg_bb )
                L_bkg_bb = cell((B.band_count()-1),1); % cache for local adaptation
            end
            switch metric_par.local_adapt
                case 'simple'
                    if isempty( L_bkg_bb{bb} )
                        L_bkg_bb{bb} = max( 1e-4, imresize( L_adapt, size(T_f), 'bicubic' ) );
                    end
                    L_bkg = L_bkg_bb{bb};
                case 'gpyr' % Use the Gaussian pyramid at the level b+1
                    if isempty( L_bkg_bb{bb} )
                        bnd = B.get_gauss_exp_band(bb+1);
                        gauss_band = bnd(:,:,2); % always use reference, sustained
                        if strcmp( metric_par.contrast, 'log' )
                            gauss_band = 10.^gauss_band;
                        end
                        L_bkg_bb{bb} = gauss_band; %gausspyr_expand( gauss_band, size(R_f) );
                    end
                    L_bkg = L_bkg_bb{bb};
                case 'gpyr2' % Use the Gaussian pyramid at the level b+2
                    if isempty( L_bkg_bb{bb} )
                        use_band = min(bb+2, B.height);
                        bnd = B.get_gauss_band(use_band);
                        gauss_band = bnd(:,:,2); % always use reference, sustained
                        if strcmp( metric_par.contrast, 'log' )
                            gauss_band = 10.^gauss_band;
                        end
                        for br = (use_band-1):-1:bb
                            band_size = B.band_size(br,0);
                            gauss_band = gausspyr_expand(gauss_band, band_size([1 2]));
                        end
                        L_bkg_bb{bb} = gauss_band;
                    end
                    L_bkg = L_bkg_bb{bb};
                case 'gpyr0' % Use the same level of the Gaussian pyramid
                    if isempty( L_bkg_bb{bb} )
                        bnd = B.get_gauss_band(bb);
                        gauss_band = bnd(:,:,2); % always use reference, sustained
                        if strcmp( metric_par.contrast, 'log' )
                            gauss_band = 10.^gauss_band;
                        end
                        L_bkg_bb{bb} = gauss_band;
                    end
                    L_bkg = L_bkg_bb{bb};
                case 'global' % Global adaptation - one value per image
                    L_bkg = ones(size(T_f), 'like', T_f) * L_adapt;
                otherwise
                    error( 'Unknown type of local adaptation' );
            end
            
            if ~strcmp( metric_par.contrast, 'log' )
                % Weber contrast
                max_contrast = 1000; % The contrast that exceed a certain value may lead to numerical issues
                T_f = min(T_f./L_bkg, max_contrast);
                R_f = min(R_f./L_bkg, max_contrast);
            end
            
            if isempty( N_nCSF{bb,cc} ) % need to precompute the CSF (the inverse of)
                
                if metric_par.do_foveated % Fixation, parafoveal sensitivity
                    if size(metric_par.fixation_point,1)>1 % moving fixation point
                        fix_point = metric_par.fixation_point(ff,:)+1;
                    else
                        fix_point = metric_par.fixation_point+1;
                    end
                    
                    if isempty( metric_par.content_mapping )

                        xv = single(0:(size(T_f,2)-1));
                        yv = single(0:(size(T_f,1)-1));
                        if metric_par.use_gpu
                            xv = gpuArray( xv );
                            yv = gpuArray( yv );
                        end
                        [xx, yy] = meshgrid( xv, yv );
                        df = video_sz(2)/size(T_f,2); % Downscale factor
                    
                        ecc = sqrt( (xx-fix_point(1)/df).^2 + (yy-fix_point(2)/df).^2 )*(df/pix_per_deg);
                    else
                        df = video_sz(2)/size(T_f,2); % Downscale factor
                        ecc = metric_par.content_mapping.get_eccentricity_map( [size(T_f,1) size(T_f,2)], fix_point/df );
                        if metric_par.use_gpu
                            ecc = gpuArray( single(ecc) );
                        end
                    end
                    %ecc = zeros(size(L_bkg));
                    rho = rho_band(bb).*display_geometry.get_resolution_magnification(ecc);
                else % No fixation, foveal sensitivity everywhere
                    if metric_par.use_gpu                    
                        rho = gpuArray(single(rho_band(bb)));
                        ecc = gpuArray(single(0));
                    else
                        rho = single(rho_band(bb));
                        ecc = single(0);
                    end
                end
                
                S = csf.sensitivity_cached( rho, omega(cc), L_bkg, ecc, metric_par.csf_sigma, metric_par.k_cm ) * 10^(metric_par.sensitivity_correction/20);                
                
                if strcmp( metric_par.contrast, 'log' )
                    N_nCSF{bb,cc} = weber2log( min( 1./S, 0.9999999 ) );
                else
                    N_nCSF{bb,cc} = 1./S;
                end
            end
            
            if ~isempty( metric_par.band_callback )
                metric_par.band_callback( ff, N, bb, (B.band_count()-1), cc, T_f, R_f, N_nCSF{bb,cc} );
            end
            
            D = masking_model( metric_par, T_f, R_f, N_nCSF{bb,cc}, cc );
            
            if ~isempty( metric_par.content_mapping ) % Relevant only for 360 videos
                % To make sure that we cannot see anything behind our head                
                % 105 deg is the maximum horizontal FOV - a bit
                % conservative assumption
                D(ecc>105) = 0; 
            end
            
            if( metric_par.debug ) % For debugging
                %D_v = D;
                %D_v(D<1) = 0;
                D_dd = D_dd.set_band(bb,1,D);
            end
                        
            if metric_par.do_diff_map
                if cc==1
                    Dmap_pyr = Dmap_pyr.set_band(bb,1,D);
                else
                    Dmap_pyr = Dmap_pyr.set_band(bb,1, Dmap_pyr.get_band(bb,1)+w_temp_ch(cc)*D);
                end
            end
            
            if isempty( Q_per_ch )
                % Note that we always store the results for two temporal
                % channels. For images, trasient response = 0.
                Q_per_ch = zeros((B.band_count()-1), 2, N, 'like', D);
            end
            % Per-band error used for quality predictions
            Q_band = w_temp_ch(cc)*lp_norm( D(:), metric_par.beta, 1, true );
            Q_per_ch(bb,cc,ff) = Q_band;
            %Q_err(ff) = Q_err(ff) + Q_band;
        end
        
        if( metric_par.debug ) % For debugging            
            D_debug{cc}(:,:,ff) = gather(imresize(w_temp_ch(cc)*D_dd.reconstruct(), vid_size(1:2), 'bicubic' ));
        end
        if metric_par.do_diff_map
            Dmap(:,:,ff) = gather(Dmap_pyr.reconstruct());
        end
        
    end
    
%     gpu = gpuDevice;
%     fprintf( 1, 'GPU mem: %g MB\n', gpu.AvailableMemory / 1e6 );
    
end


res = struct();
% We need to clamp to a large value as very large differences can cause
% Infs
%res.Q = gather( min( 1e8, minkowski_sum(Q_err,metric_par.beta_temp) ) );

% Integrate first across spatial frequencies (no normlization), than across temporal
% channels (no normalization), then across the frames (normalized)
Q_sc = lp_norm( Q_per_ch, metric_par.beta_sch, 1, false );
Q_tc = lp_norm( Q_sc, metric_par.beta_tch, 2, false );
res.Q = gather( lp_norm( Q_tc, metric_par.beta_t, 3, true ) );

%res.Q_per_ch = gather((Q_per_ch/N).^(1/metric_par.beta_temp));

if( metric_par.debug ) % For debugging
    clf
    hold on
    hh(1) = plot( squeeze(Q_sc(1,1,:)), 'DisplayName', 'Sustained' );
    hh(2) = plot( squeeze(Q_sc(1,2,:)), 'DisplayName', 'Transient' );
    hh(3) = plot( squeeze(Q_tc), 'DisplayName', 'Both' );
    hold off
    xlabel( 'Frame' );
    ylabel( 'Visual difference' );
    legend( hh, 'Location', 'best' );
    
    vid_small = zeros( vid_size(1), vid_size(2)*2, N );
    for kk=1:N
        vid_small(:,:,kk) = gather(imresize( (max(0,cat( 2, video_source.get_test_frame(kk, false), video_source.get_reference_frame(kk, false)))/100).^(1/2.2), [vid_size(1) vid_size(2)*2], 'bicubic'));
    end
    
    %max_sust = prctile( cat( 1, D_debug{1}(:), D_debug{2}(:)), 99 );
    max_sust = max( cat( 1, D_debug{1}(:), D_debug{2}(:)) );
    
    save_as_video( cat( 1, vid_small, ...
        (abs(cat( 2, D_debug{1}, D_debug{2})/max_sust)).^(1/4) ), metric_par.video_name );
    
    
end

if ~isempty( metric_par.jod_a )
    beta_jod = 10.^metric_par.log_jod_exp;
    %res.Q_JOD = -(metric_par.jod_a*res.Q).^beta_jod;
    %res.Q_JOD = metric_par.jod_a*res.Q.^beta_jod;
    res.Q_JOD = sign(metric_par.jod_a)*(abs(metric_par.jod_a)^(1/beta_jod)*res.Q).^beta_jod + 10; % This one can help with very large numbers
end

if metric_par.do_diff_map
    if ~isempty( metric_par.jod_a )
        beta_jod = 10.^metric_par.log_jod_exp;
        res.diff_map = Dmap.^beta_jod * abs(metric_par.jod_a);                    
    else
        warning( 'JOD regression data is missing' );
        res.diff_map = Dmap;                    
    end
end

end

% function y = safe_log( x )
% 
% Y_min = 1e-6; % Minimum allowed values (avoids NaNs and -Infs)
% 
% y = log10( max(x, Y_min) );
% 
% end

function d = lp_norm( X, p, dim, normalize )
% LP-norm on the vector X, using the exponent p and across the dimension
% dim. If normalize=true, then divided by the number of elements (true by
% default).

if ~exist( 'dim', 'var' )
    dim = 1;
end
if ~exist( 'normalize', 'var' ) || normalize
    N = size(X,dim);
else
    N = 1;
end

d = sum( abs(X).^p / N, dim ).^(1/p);

end


function D = masking_model( metric_par, T, R, N, cc )
% T - test frame band
% R - reference frame band
% N - detection threshold (or neural noise)
% cc - temporal channel: 1 - sustained, 2 - transient

switch metric_par.masking_model
    case 'contrast_difference' % The simplest difference, no CSF
        D = abs(T-R)/0.05;
        
    case 'contrast_difference_perc_norm' % Perceptually normalized contrast difference
        D = abs(T-R)./N;        
        
    case 'min_mutual_masking_perc_norm2'
        p = metric_par.mask_p;
        if cc==1
            q = metric_par.mask_q_sust;
        else
            q = metric_par.mask_q_trans;
        end        
        T = T./N;
        R = R./N;
        M = phase_uncertainty( min( abs(T), abs(R) ) );
        D = mask_func_perc_norm2( abs(T-R), M, p, q );
                
    otherwise
        error( 'Unknown masking model' );
        
end

D = min( D, 1e4 ); % This prevents Infs when beta is pooling is high



    function M_pu = phase_uncertainty( M )
        if metric_par.pu_dilate~=0
            M_pu = imgaussfilt( M, metric_par.pu_dilate ) * 10^metric_par.mask_c;
        else
            M_pu = M * 10^metric_par.mask_c;
        end
    end


    function R = mask_func_perc_norm2( G, G_mask, p, q )
        % Masking on perceptually normalized quantities (as in Daly's VDP)
                
        R = G.^p ./ ( 1 + (G_mask).^q ) ;
    end

end

