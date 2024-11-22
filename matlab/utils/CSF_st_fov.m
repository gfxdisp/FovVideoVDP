classdef CSF_st_fov
    % A cached spatio-temporal CSF with foveatation and luminance-dependence
    
    properties( Constant )
        Y_min = 0.001;  % The minimum luminance
        Y_max = 10000;  % The maximum luminance
        rho_min = 2^-4  % The minimum spatial frequency
        rho_max = 64;   % The maximum spatial frequency
        ecc_max = 120;  % The maximum eccentricity
        
        csf_cache_dir = fullfile( tempdir, 'csf_cache' );
        use_file_cache = true;
        semkey=1234;
    end            
    
    properties
        cache = struct();
        use_gpu = true;
    end
    
    methods
        
        function obj = CSF_st_fov( )            
            %semaphore('create',obj.semkey,1);
        end
        
        % Use a precomputed 3D LUT to compute the sensitivity
        function S = sensitivity_cached( obj, rho, omega, L_bkg, ecc, sigma, k_cm )
            
            if ~exist( 'k_cm', 'var' )
                k_cm = 1;
            end
            
            key = strrep(strrep(sprintf( 'o%g_s%g_cm%g_gpu%d', omega, sigma, k_cm, obj.use_gpu ), '-', 'n'), '.', '_'); % key into the cache
            if ~isfield( obj.cache, key )
                
                fname = fullfile( CSF_st_fov.csf_cache_dir, strcat( key, '.mat' ) );
                                
                %fname_lock = strcat( fname, '.lock' );
                
                if obj.use_file_cache && isfile( fname )
                    
                    
                    % Wait up to 1 sec for the lock to be released
                    %for ww=1:100 
                    %    if ~exist( fname_lock, 'file' )
                    %        break;
                    %    end
                    %    pause( 0.01 );
                    %end
                    %semaphore('wait',obj.semkey);                    
                    % Load LUT from file if exists
                    lt = load( fname );
                    %semaphore('post',obj.semkey);                    
                    lut = lt.lut;
                else
                    lut = obj.precompute_lut( omega, sigma, k_cm );
                    
                    if obj.use_file_cache
                        % Save LUT for later use
                        if ~isfolder( CSF_st_fov.csf_cache_dir )
                            mkdir( CSF_st_fov.csf_cache_dir );
                        end
                        
                        %fh_lock = fopen( fname_lock, 'wb' );
                        %fprintf( fh_lock, '1' );
                        %fclose( fh_lock );
                        
                        %semaphore('wait',obj.semkey);
                                                
                        tmp_fname = [tempname() '.mat'];
                        save( tmp_fname, 'lut' );
                        if ~isfile( fname )
                            movefile( tmp_fname, fname, 'f' ); % hopefully, this one will be atomic
                        end
                        %semaphore('post',obj.semkey);
                        %delete( fname_lock );
                    end
                    
                end
                
                obj.cache.(key).lut = lut;
            else
                lut = obj.cache.(key).lut;
            end
            
            Y_q = log2(clamp(L_bkg,lut.Y(1), lut.Y(end)));
            
            if numel(rho)==1 % if rho is a scalar
                rho_q = ones(size(Y_q), 'like', rho ) * log2(clamp(rho, lut.rho(1), lut.rho(end)));
            else
                rho_q = log2(clamp(rho,lut.rho(1), lut.rho(end)));
            end
            if numel(ecc)==1 % if ecc is a scalar
                ecc_q = ones(size(Y_q), 'like', ecc ) * sqrt(clamp(ecc, lut.ecc(1), lut.ecc(end)));
            else
                ecc_q = sqrt(clamp(ecc, lut.ecc(1), lut.ecc(end)));
            end
            
            % Note that to confuse everyone, the first two dimensions in
            % interp3 are swapped
            S = 2.^interp3( lut.rho_log, lut.Y_log, lut.ecc_sqrt, lut.S_log, ...
                rho_q, Y_q, ecc_q, 'linear' );
            
        end

        function lut = precompute_lut( obj, omega, sigma, k_cm )
            
            N = 32;
            lut = struct;
            lut.Y = single(logspace( log10(CSF_st_fov.Y_min), log10(CSF_st_fov.Y_max), N ));
            lut.rho = single(logspace( log10(CSF_st_fov.rho_min), log10(CSF_st_fov.rho_max), N ));
            lut.ecc = single(linspace( 0, sqrt(CSF_st_fov.ecc_max), N ).^2);
            
            if obj.use_gpu
                lut.Y = gpuArray( lut.Y );
                lut.rho = gpuArray( lut.rho );
                lut.ecc = gpuArray( lut.ecc );
            end
            
            lut.Y_log = log2(lut.Y);
            lut.rho_log = log2(lut.rho);
            lut.ecc_sqrt = sqrt(lut.ecc);
            
            [Y_gd, rho_gd, ecc_gd] = ndgrid( lut.Y, lut.rho, lut.ecc );
            
            S = max( CSF_st_fov.sensitivity( rho_gd(:), omega, Y_gd(:), ecc_gd(:), sigma, k_cm ), 1e-4);
            
            lut.S_log = reshape( log2(S), length(lut.Y), length(lut.rho), length(lut.ecc) );
        end
        
    end
    
    methods( Static )
        
        
        function S = sensitivity( rho, omega, L_bkg, ecc, sigma, k_cm )
            % Spatio-temporal CSF with foveatation and luminance-dependence
            %
            % rho - sparial frequency in cyces per degree (scalar)
            % omega - temporal frequence in Hz (scalar)
            % L_bkg - background luminance in cd/m^2 (matrix)
            % ecc - eccentricity in deg (matrix)
            
            if ~exist( 'sigma', 'var' )
                sigma = 1;
            end
            
            if sigma<0
                % Fixed cycles condition, sigma represents a wavelength
                sigma = -sigma./rho;
            end

            M_rel = (cortical_magnification_dougherty(ecc)/cortical_magnification_dougherty(0)).^k_cm;

            %M_rel = k_cm * cortical_magnification_dougherty(0)./cortical_magnification_dougherty(ecc);
            
            A_cm = pi*(sigma.*M_rel).^2; % Stimulus size adjusted for coortical magnification
            rho_cm = rho./M_rel; % Frequency adjusted for coortical magnification
            
            % The relative change in sensitivity due to temporal frequency
            S_st = csf_spatiotemp_daly( rho_cm, omega )./(csf_spatiotemp_daly( rho_cm, 0 ) + 1e-5);
            
            % Kelly’s CSF was measured at 300 Td (2.3mm pupil) = 72 cd/m^2
            %L_st = 72;
            
            sccsf = SCCSF_ConeContrastMat();
            LMS_d65 = xyz2lms2006( whitepoint( 'd65' ) );
            % Achromatic sensitivity, accounting for luminance, frequency and size
            S_sp = sccsf.sensitivity_coldir( rho_cm, L_bkg(:) * LMS_d65, 1, A_cm );
            
            S = reshape( S_sp .* S_st, size(L_bkg) );
            
        end
        
    end
    
end