classdef SCCSF
    %Super class of all Spatio-chromatic CSF models
    
    properties
        par; % Model parameters as a structure
    end
    
    methods( Abstract )
        
        % A short name that could be used as a part of a file name
        name = short_name( obj )
        
        % Predict the probility of detecting a Gabour patch of certain chromatic
        % direction and amplitide
        %
        % [P, C] = camliv_colour_difference( freq, LMS_mean, LMS_delta, area, params )
        %
        % freq - spatial frequency in cpd
        % LMS_mean - LMS of the background colour (CIE2006 CMF)
        % LMS_delta - colour direction vector in the LMS space (LMS_peak-LMS_mean)
        % area - area in deg^2
        % params - fitted parameters
        %
        % The method returns:
        % P - The probability of detection
        % C - Normalized detection contrast (1 when P=0.5)
        [P, C] = pdet( obj, freq, LMS_mean, LMS_delta, area )
        
        % Predict the sensitivity for a detection of a Gabour patch of certain chromatic
        % direction and amplitide. The difference between this method
        % and pdet() is that it predicts where the threshold is
        % (LMS_delta_thr) while pdet() onlt tells what is the
        % probability of detection for a given LMS_delta.
        %
        % [P, C, S, LMS_delta_thr] = camliv_colour_difference( freq, LMS_mean, LMS_delta, area, params )
        %
        % freq - spatial frequency in cpd
        % LMS_mean - LMS of the background colour (CIE2006 CMF)
        % LMS_delta - colour direction vector in the LMS space (LMS_peak-LMS_mean)
        % area - area in deg^2
        % params - fitted parameters
        %
        % The method returns:
        % S - Sensitivity (the inverse of cone contrast at the threshold)
        % LMS_delta_thr - vector of the same direction as LMS_delta, but with the
        %     length adjusted so that it points to the detection threshold
        % P - The probability of detection
        % C - Normalized detection contrast (1 when P=0.5)
        [S, LMS_delta_thr, P, C] = sensitivity( obj, freq, LMS_mean, LMS_delta, area )
        
    end
    
    methods
        
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
        
        
        function [S, LMS_delta_thr, P, C] = sensitivity_coldir( obj, freq, LMS_mean, color_direction, area )
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
                        [s(cc).(ff{kk}), pos_ret] = SCCSF.param2struct(s(cc).(ff{kk}), pars_vector(pos:end) );
                        pos = pos+pos_ret-1;
                    else
                        N = length(s(cc).(ff{kk}));
                        s(cc).(ff{kk}) = pars_vector(pos:(pos+N-1));
                        pos = pos + N;
                    end
                end
            end
        end
        
        
        function pars_vector = struct2param( s )
            % Convert a structure to a vector with the parameters
            
            pars_vector = [];
            
            for cc=1:length(s)
                ff = fieldnames(s(cc));
                for kk=1:length(ff)
                    
                    if isstruct( s(cc).(ff{kk}) )
                        pars_vector = cat( 2, pars_vector, SCCSF.struct2param(s(cc).(ff{kk})) );
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
                    v = 10.^(pars(1)*log_lum + pars(2));
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
            
            % These multipliers are used to adjust the sensitivity of each dataset by a
            % small factor
            p = struct();
            p.ds.xuqiang = 0.914029;
            p.ds.sw = 0.861676;
            p.ds.kim2013_ach = 0.817432;
            p.ds.kim2013_ch = 0.964859;
            p.ds.four_centres = 1;
            
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

