classdef hdrvdp_lpyr_dec < hdrvdp_multscale
    % Decimated laplacian pyramid
    
    properties
        P;
        P_gauss;
        P_gauss_exp;
        
        ppd;
        base_ppd;
        img_sz;
        band_freqs;
        height;
        do_gauss = false;
        do_gauss_exp = false;
        min_freq = 0.5;
    end
    
    methods

        % Create an empty image pyramid
        % sz - size of the original image        
        function ms = create( ms, sz, ppd )
            ms.ppd = ppd;
            ms.img_sz = sz;

            ms.height = ms.get_max_height( sz );
            
            % Frequency peaks of each band
            ms.band_freqs = [1 0.3228*2.^-(0:(ms.height-1))] * ms.ppd/2;                        
          
%             ms.P = cell(ms.height,1);
%             lev_sz = sz;
%             for kk=1:ms.height
%                 ms.P{kk} = zeros(lev_sz);
%                 lev_sz = floor( lev_s/2 );
%             end
            
        end
        
        function ms = decompose( ms, I, ppd )
            
            ms.ppd = ppd;
            ms.img_sz = size(I);
                        
            ms.height = ms.get_max_height( size(I) );
            
            % Frequency peaks of each band
            ms.band_freqs = [1 0.3228*2.^-(0:(ms.height-1))] * ms.ppd/2;                        
                            
            if ms.do_gauss_exp && ms.do_gauss
                [ms.P, ms.P_gauss, ms.P_gauss_exp] = laplacian_pyramid_dec( I, ms.height+1 );
            elseif ms.do_gauss_exp 
                [ms.P, ~, ms.P_gauss_exp] = laplacian_pyramid_dec( I, ms.height+1 );
            elseif ms.do_gauss
                [ms.P, ms.P_gauss] = laplacian_pyramid_dec( I, ms.height+1 );
            else
                ms.P = laplacian_pyramid_dec( I, ms.height+1 );
            end
        end

        function height = get_max_height( ms, sz )
            
            % The maximum number of levels we can have
            max_levels = floor(log2(min(sz(1),sz(2))))-1;
            
            % We want the minimum frequency the band to be min_freq
            max_band = find( [1 0.3228*2.^-(0:14)] * ms.ppd/2 <= ms.min_freq, 1 );
            if isempty(max_band)
                max_band=max_levels;
            end
            
            height = clamp( max_band, 1, max_levels );
        end
        
        function I = reconstruct( ms )
            
            I = ms.P{end};
            for i=(length(ms.P)-1):-1:1
                I = gausspyr_expand( I, [size(ms.P{i},1) size(ms.P{i},2)] );
                I = I + ms.P{i};
            end
            
        end
        
        function B = get_band( ms, band, o )
            
            if band == 1 || band == length(ms.P)
                band_mult = 1;
            else
                band_mult = 2;
            end
            
            B = ms.P{band} * band_mult;
        end

        function B = get_gauss_band( ms, band, o )
            
            if ~ms.do_gauss
                error( 'do_gauss property needs to be set to true' );
            end
                        
            B = ms.P_gauss{band};
        end

        function B = get_gauss_exp_band( ms, band, ~ )
            % Get the level of the Gaussian pyramid that has been expanded
            % to the resolution of the lower (higher resolution) band. 
            % Those are produced as a by-produced of creating Gaussian
            % pyramid.
            
            if ~ms.do_gauss_exp
                error( 'do_gauss_exp property needs to be set to true' );
            end
                        
            B = ms.P_gauss_exp{band};
        end
        
        function ms = set_band( ms, band, o, B )
            
            if band == 1 || band == length(ms.P)
                band_mult = 1;
            else
                band_mult = 2;
            end
            
            ms.P{band} = B/band_mult;
        end
        
        function bc = band_count( ms )
            bc = length(ms.P);
        end
        
        function oc = orient_count( ms, band )
            oc = 1;
        end
        
        function sz = band_size( ms, band, o )
            sz = size( ms.P{band} );
        end
        
        function bf = get_freqs( ms )
            bf = ms.band_freqs;
        end
        
    end
    
end