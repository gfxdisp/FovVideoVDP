classdef CSF_base
    %Super class of all Spatio-chromatic CSF models
    
    properties
        par; % Model parameters as a structure
    end
    
    methods( Abstract )
        
        % A short name that could be used as a part of a file name
        name = short_name( obj )
                

        % A general interface to compute sensitivity, which takes a
        % structure with the parameters values. This allows to add/remove
        % parameters without changing the interface, more flexible use of
        % parameters (e.g. either LMS or luminance of the background) and
        % should be less error prone. 
        % 
        % pars is the structure with the field names that closely match
        % the column names in the data files:
        %
        % luminance - luminance in cd/m^2. D65 background is assumed. 
        % lms_bkg - specify background colour and luminance 
        % s_frequency - spatial frequency in cpd
        % t_frequency - temporal frequency in Hz
        % orientation - orientation in deg (default 0)
        % lms_delta - modulation direction in the lms colour space (default
        %             D65 luminance modulation)
        % area - area of the stimulus in deg^2
        % ge_sigma - radius of the gaussian envelope in deg 
        % eccentricity - eccentricity in deg (default 0)       
        % vis_field - orientation in the visual field. See README.md
        %             (default 0)
        %
        % 'lms_bkg' and 'lms_delta' should be either [1 3] or [N 3] matrix. 
        % Any other field should be a column vector of size N or a scalar. 
        %
        % You must specify 'luminance' or 'lms_bkg' but not both. 
        % You must specify 'area' or 'ge_sigma' but not both. 
        %
        % Example: 
        % csf_pars = struct( 's_frequency', pars.s_freq, 't_frequency', pars.t_freq, 'orientation', pars.orientation, 'lms_bkg', LMS_mean, 'lms_delta', LMS_delta, 'area', area, 'eccentricity', pars.eccentricity );          
        % S = csf_model.sensitivity( csf_pars );
        S = sensitivity( obj, pars );
        
        %S = sensitivity_ach( obj, freq, luminance, area )
        
    end
    
    methods
        
        function str = full_name( obj )
            str = obj.short_name();
        end
        
        function obj = set_pars( obj, pars_vector )
            % Set the parameters of the model, supplied as a row vector
            % (used for optimizing the parameters of the model)
            
            assert( ~isempty( obj.par ) ); % obj.par must be initialized before calling this function
            
            obj.par = obj.param2struct( obj.par, pars_vector );
        end
        
        function pars_vector = get_pars( obj )
            % Get the parameters of the model as a row vector
            % (used for optimizing the parameters of the model)
            
            pars_vector = obj.struct2param( obj.par );
        end
        
        % Predict the sensitivity for a detection of a Gabour patch of certain chromatic
        % direction and amplitide. 
        %
        % S = sensitivity_stolms( obj, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity );
        %
        % Important: all parameters must be column vectors. LMS_bkg and
        %            LMS_delta are Nx3 matrices
        %
        % freq - spatial frequency in cpd
        % LMS_bkg - LMS of the background colour (CIE2006 CMF)
        % LMS_delta - colour direction vector in the LMS space (LMS_peak-LMS_mean)
        % area - area in deg^2
        %
        % The method returns:
        % S - Sensitivity (the inverse of cone contrast at the threshold)
        function S = sensitivity_stolms( obj, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity )
            csf_pars = struct( 's_frequency', s_freq, 't_frequency', t_freq, 'orientation', orientation, 'lms_bkg', LMS_bkg, 'lms_delta', LMS_delta, 'area', area, 'eccentricity', eccentricity );
            S = obj.sensitivity( csf_pars );
        end        

        function S = sensitivity_stolmsv( obj, s_freq, t_freq, orientation, LMS_bkg, LMS_delta, area, eccentricity, vis_field )
            csf_pars = struct( 's_frequency', s_freq, 't_frequency', t_freq, 'orientation', orientation, 'lms_bkg', LMS_bkg, 'lms_delta', LMS_delta, 'area', area, 'eccentricity', eccentricity, 'vis_field', vis_field );
            S = obj.sensitivity( csf_pars );
        end        
        
        % Test whether all the parameters are correct size, that the
        % names are correct, set the default values for the missing
        % parameters. 
        % pars - the csf_params structure
        % requires - a cell array with the selected parameters that are
        %            required. Used for the multually exclusive
        %            parameters, such as 'luminance'/'lms_bkg' and 
        %            'area'/'ge_sigma' so that one of them is computed as
        %            needed, but not both.
        % expand - if true, all the parameters are expanded to have the
        %          same size.
        function pars = test_complete_params(obj, pars, requires, expand )

            if ~exist( 'expand', 'var' )
                expand = false;
            end

            valid_names = { 'luminance', 'lms_bkg', 'lms_delta', 's_frequency', 't_frequency', 'orientation', 'area', 'ge_sigma', 'eccentricity', 'vis_field' };            
            %par_len = [1 3 3 1 1 1 1 1 1]; % Vector size for each argument
            fn = fieldnames( pars );
            N = 1; % The size of the vector
            for kk=1:length(fn)
                if ~ismember( fn{kk}, valid_names )
                    error( 'Parameter structure contains unrecognized field ''%s''', fn{kk} );
                end
                if ismember( fn{kk}, { 'lms_bkg', 'lms_delta' } )
                    par_len = 3;
                else
                    par_len = 1;
                end
                % Check whether the size is correct
                if numel(pars.(fn{kk})) > 1
                    Nc = numel(pars.(fn{kk}))/par_len;
                    if N==1
                        N = Nc;
                    else
                        if Nc~=1 && N ~= Nc
                            error( 'Inconsistent size of the parameter ''%s''', fn{kk} );
                        end
                    end
                end
            end

            if ismember( 'luminance', requires )
                if ~isfield( pars, 'luminance')
                    if ~isfield( pars, 'lms_bkg')
                        error( 'You need to pass either luminance or lms_bkg parameter.')
                    end
                    pars.luminance = pars.lms_bkg(:,1) + pars.lms_bkg(:,2);                    
                end
            end

            if ismember( 'lms_bkg', requires )
                if ~isfield( pars, 'lms_bkg')
                    if ~isfield( pars, 'luminance')
                        error( 'You need to pass either luminance or lms_bkg parameter.')
                    end
                    pars.lms_bkg = [0.6991 0.3009 0.0198] .* pars.luminance;
                end
            end

            if ismember( 'ge_sigma', requires )
                if ~isfield( pars, 'ge_sigma')
                    if ~isfield( pars, 'area')
                        error( 'You need to pass either ge_sigma or area parameter.')
                    end
                    pars.ge_sigma = sqrt(pars.area/pi);
                end
            end

            if ismember( 'area', requires )
                if ~isfield( pars, 'area')
                    if ~isfield( pars, 'ge_sigma')
                        error( 'You need to pass either ge_sigma or area parameter.')
                    end
                    pars.area = pi*pars.ge_sigma.^2;
                end
            end
            
            % Default parameter values
            def_pars = struct( 'eccentricity', 0, 'vis_field', 0, 'orientation', 0, 't_frequency', 0, 'lms_delta', [0.6855 0.2951 0.0194] );
            fn_dp = fieldnames( def_pars );
            for kk=1:length(fn_dp)
                if ~isfield(pars, fn_dp{kk})
                    pars.(fn_dp{kk}) = def_pars.(fn_dp{kk});
                end
            end
            
            if expand && N>1
                % Make all parameters the same height 
                fn = fieldnames( pars );
                for kk=1:length(fn)
                    if size(pars.(fn{kk}),1)==1
                        pars.(fn{kk}) = repmat( pars.(fn{kk}), [N 1]);
                    end
                end                
            end            

        end
        
        function [s_freq, t_freq, orientation, LMS_mean, LMS_delta, area, eccentricity] = test_and_expand_pars(obj, s_freq, t_freq, orientation, LMS_mean, LMS_delta, area, eccentricity)
            
            N = max( [size(s_freq,1) size(t_freq,1) size(orientation,1) size(LMS_mean,1) size(LMS_delta,1) size(area,1) size(eccentricity,1)] );
            
            if any(size(s_freq)~=1) && any(size(s_freq)~=[N 1])
                error( '''s_freq'' must be a column vector or a scalar' );
            end

            if any(size(t_freq)~=1) && any(size(t_freq)~=[N 1])
                error( '''t_freq'' must be a column vector or a scalar' );
            end
            
            if any(size(orientation)~=1) && any(size(orientation)~=[N 1])
                error( '''orientation'' must be a column vector or a scalar' );
            end

            if any(size(LMS_mean)~=[1 3]) && any(size(LMS_mean)~=[N 3])
                error( '''LMS_mean'' must be a [N x 3] matrix' );
            end

            if any(size(LMS_delta)~=[1 3]) && any(size(LMS_delta)~=[N 3])
                error( '''LMS_delta'' must be a [N x 3] matrix' );
            end

            if any(size(area)~=1) && any(size(area)~=[N 1])
                error( '''area'' must be a column vector or a scalar' );
            end

            if any(size(eccentricity)~=1) && any(size(eccentricity)~=[N 1])
                error( '''eccentricity'' must be a column vector or a scalar' );
            end
            
            if N>1
                % Expand scalars into the vectors of the same size
                
                if size(s_freq,1) == 1
                    s_freq = repmat( s_freq, [N 1] );
                end
                if size(t_freq,1) == 1
                    t_freq = repmat( t_freq, [N 1] );
                end
                if size(orientation,1) == 1
                    orientation = repmat( orientation, [N 1] );
                end
                if size(LMS_mean,1) == 1
                    LMS_mean = repmat( LMS_mean, [N 1] );
                end
                if size(LMS_delta,1) == 1
                    LMS_delta = repmat( LMS_delta, [N 1] );
                end
                if size(area,1) == 1
                    area = repmat( area, [N 1] );
                end
                if size(eccentricity,1) == 1
                    eccentricity = repmat( eccentricity, [N 1] );
                end
                if size(orientation,1) == 1
                    orientation = repmat( orientation, [N 1] );
                end
                if size(orientation,1) == 1
                    orientation = repmat( orientation, [N 1] );
                end
                
            end
            
        end
        
        function [S, LMS_delta_thr, P, C, LMS_delta] = sensitivity_coldir( obj, freq, LMS_mean, color_direction, area )
            % Sensitivity measured along one of the colour directions in
            % the DKL colour space.
            %
            % [S, LMS_delta_thr, P, C] = sensitivity_coldir( obj, freq, LMS_mean, color_direction, area )
            %
            
            switch color_direction
                case 1 % achromatic (luminance)
                    LMS_delta = [0.698 0.302 0.019613];
                case 2 % red-green
                    LMS_delta = [0.302 -0.302 0];
                case 3 % violet-yellow
                    LMS_delta = [0 0 0.019613];
                otherwise
                    error( 'color_direction must be 1, 2 or 3' );
            end
            
            [S, LMS_delta_thr, P, C] = obj.sensitivity( freq, LMS_mean, LMS_delta, area );
            
        end
        
       function print( obj, fh )
            % Print the model parameters in a format ready to be pasted into
            % get_default_par()
            
            obj.print_struct( fh, 'p.', obj.par );            
        end
         
        function print_struct( obj, fh, struct_name, s )
            % Print the model parameters in a format ready to be pasted into
            % get_default_par()
            
            fn = fieldnames( s );
            for ff=1:length(fn)
                if ismember( fn{ff}, { 'cm', 'ds' } )
                    continue;
                end
                if isstruct(s.(fn{ff}))
                    obj.print_struct( fh, strcat( struct_name, fn{ff}, '.' ), s.(fn{ff}) );
                else
                    fprintf( fh, '\t%s%s = ', struct_name, fn{ff} );
                    obj.print_vector( fh, s.(fn{ff}) );
                    fprintf( fh, ';\n' );
                end
            end
            
        end
        
        function test_sensitivity_predictions( obj )
            % Test whether sensitivity() method correctly predicts the
            % threshold. It a unit test to prevent mistakes in the
            % computation of the thresholds. The function runs a binary
            % search to numerically estimate the threshold and compares
            % with the sensitivity() method estimate.
                        
            freq = 2;
            area = 0.5^2*pi;
            epsilon = 1e-4;
            
            Ys = [0.1 1 10 100];
            
            for kk=1:length(Ys)
                Y_mean = Ys(kk);
                LMS_mean = xyz2lms2006( whitepoint( 'd65' )*Y_mean );
                LMS_delta = (rand( [1 3] )*2-1) * 0.01*Y_mean;
                
                C_thr = 1.0;
                
                thr = binary_search( @bs_func, C_thr, [0.001, 100], 40);
                [~, LMS_delta_thr] = obj.sensitivity( freq, LMS_mean, LMS_delta, area );
                
                thr_s = LMS_delta_thr./LMS_delta;
                
                % The predicted LMS_delta_thr must have the same direction as
                % LMS_delta
                assert( abs(thr_s(1)-thr_s(2))<epsilon && abs(thr_s(2)-thr_s(3))<epsilon );
                
                % Binary search should result is almost the same threshold as
                % sensitivity() method.
                assert( (thr_s(1)-thr) < epsilon );
                
            end
            
            function C = bs_func( scalar )
                
                [~,C] = obj.pdet( freq, LMS_mean, LMS_delta*scalar, area );
            end
                        
        end
        
    end
    
    methods(Static)
        function [s, pos] = param2struct( s, pars_vector )
            % Convert a vector with the parameters to a structure with the
            % same fields as in the structure "s".
            
            pos = 1;
            for cc=1:length(s)
                ff = fieldnames(s(cc));
                for kk=1:length(ff)
                    
                    if isstruct(s(cc).(ff{kk}))
                        [s(cc).(ff{kk}), pos_ret] = CSF_base.param2struct(s(cc).(ff{kk}), pars_vector(pos:end) );
                        pos = pos+pos_ret-1;
                    else
                        N = length(s(cc).(ff{kk}));
                        s(cc).(ff{kk}) = pars_vector(pos:(pos+N-1));
                        pos = pos + N;
                    end
                end
            end
%             if (pos-1) ~= length(pars_vector)
%                 error( 'The parameter vector contains %d elements while the model has %d optimized parameters. Perhaps the optimized for a different set of datasets?', length(pars_vector), (pos-1) );
%             end
        end
        
        
        function pars_vector = struct2param( s )
            % Convert a structure to a vector with the parameters
            
            pars_vector = [];
            
            for cc=1:length(s)
                ff = fieldnames(s(cc));
                for kk=1:length(ff)
                    
                    if isstruct( s(cc).(ff{kk}) )
                        pars_vector = cat( 2, pars_vector, CSF_base.struct2param(s(cc).(ff{kk})) );
                    else
                        pars_vector = cat( 2, pars_vector, s(cc).(ff{kk}) );
                    end
                    
                end
            end
            
        end
        
        
        function v = get_lum_dep( pars, L )
            % A family of functions modeling luminance dependency
            
            log_lum = log10(L);
            
            switch length(pars)
                case 1
                    % Constant
                    v = ones(size(L)) * pars(1);
                case 2
                    % Linear in log
                    v = 10.^(pars(1)*log_lum + log10(pars(2)));
                case 3
                    % Log parabola
                    %        v = pars(1) * 10.^(exp( -(log_lum-pars(2)).^2/pars(3) ));
                    
                    % A single hyperbolic function
                    v = pars(1)*(1+pars(2)./L).^(-pars(3));
                case 5
                    % Two hyperbolic functions
                    v = pars(1)*(1+pars(2)./L).^(-pars(3)) .* (1-(1+pars(4)./L).^(-pars(5)));
                otherwise
                    error( 'not implemented' );
            end
            
        end

        function v = get_lum_dep_dec( pars, L )
            % A family of functions modeling luminance dependency. 
            % The same as abobe but the functions are decreasing with
            % luminance
            
            log_lum = log10(L);
            
            switch length(pars)
                case 1
                    % Constant
                    v = ones(size(L)) * pars(1);
                case 2
                    % Linear in log
                    v = 10.^(-pars(1)*log_lum + pars(2));
                case 3        
                    % A single hyperbolic function
                    v = pars(1)* (1-(1+pars(2)./L).^(-pars(3)));
                case 5
                    % Two hyperbolic functions
                    error( 'TODO' );
                    v = pars(1)*(1+pars(2)./L).^(-pars(3)) .* (1-(1+pars(4)./L).^(-pars(5)));
                otherwise
                    error( 'not implemented' );
            end
            
        end
        
        function p = get_dataset_par()
            p = struct();            
        end
        
        function print_vector( fh, vec )
            if length(vec)>1
                fprintf( fh, '[ ' );
                fprintf( fh, '%g ', vec );
                fprintf( fh, ']' );
            else
                fprintf( fh, '%g', vec );
            end
        end        
        
    end
end

