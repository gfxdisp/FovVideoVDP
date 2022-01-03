classdef CSF_temp_channels_ach < CSF_base
    % Spatio-temporal-eccentricity CSF
    % Rely on achromatic (single colour mechanism) CSF
    % 
    % Decompose into sustained and trasient channels
    % Separate CSF for both
    % Cortical magnification
    
    properties( Constant )
        Y_min = 0.001;  % The minimum luminance
        Y_max = 10000;  % The maximum luminance
        rho_min = 2^-4  % The minimum spatial frequency
        rho_max = 64;   % The maximum spatial frequency
        ecc_max = 120;  % The maximum eccentricity        
    end            
    
    properties
        use_gpu = true;
        ps_beta = 1;
    end
    
    methods
        
        function obj = CSF_temp_channels_ach( )                        
            obj.par = obj.get_default_par();                                   
        end

        function name = short_name( obj )
            % A short name that could be used as a part of a file name
            name = 'ste-csf-temp-ach';
        end
        
        function name = full_name( obj )
            name = 'steCSF temp channels achromatic';
        end

        function S = sensitivity( obj, csf_pars )
            
            csf_pars = obj.test_complete_params(csf_pars, { 'luminance', 'ge_sigma' } );

            ecc = csf_pars.eccentricity;
            sigma = csf_pars.ge_sigma;
            rho = csf_pars.s_frequency;
            omega = csf_pars.t_frequency;
            lum = csf_pars.luminance;
                        
            [R_sust, R_trans] = get_sust_trans_resp(obj, omega);
            
            M_rel = obj.rel_cortical_magnif( ecc );
            
            A_cm = pi*(sigma.*M_rel).^2; % Stimulus size adjusted for coortical magnification
            rho_cm = rho./M_rel; % Frequency adjusted for coortical magnification
                        
            pm_ratio = obj.p_to_m_ratio(ecc);
            
            S_sust = obj.csf_achrom( rho_cm, A_cm, lum, obj.par.ach_sust );
             
%             ach_trans = obj.par.ach_trans;
%             ach_trans.S_max = [1 ach_trans.S_max_sh];
%             S_trans = obj.csf_achrom( rho_cm, A_cm, lum, ach_trans );

            S_trans = obj.csf_achrom( rho_cm, A_cm, lum, obj.par.ach_trans );
            
            S_aux = obj.aux_sensitivity( csf_pars );
            if obj.ps_beta ~= 1 
                beta = obj.ps_beta;
                S = ( (R_sust.*S_sust .* sqrt(pm_ratio)).^beta + (R_trans.*S_trans .* sqrt(1./pm_ratio)).^beta + S_aux.^beta).^(1/beta);
            else
                S = R_sust.*S_sust .* sqrt(pm_ratio) + R_trans.*S_trans .* sqrt(1./pm_ratio) + S_aux;
            end

        end


        % Get the sustained and transient temporal response function
        function [R_sust, R_trans] = get_sust_trans_resp(obj, omega)
            sigma_sust = 2.5690;
            beta_sust = 1.3314;
%             sigma_trans = 0.2429;
%             beta_trans = 0.1898;            
            omega_0 = 5;

             beta_trans = 0.1898;            
             sigma_trans = obj.par.sigma_trans;

            R_sust = exp( -omega.^beta_sust / (2*sigma_sust^2) );
            R_trans = exp( -abs(omega.^beta_trans-omega_0^beta_trans).^2 / (2*sigma_trans^2) );            
        end
        
        function S = aux_sensitivity( obj, csf_pars )
            S = 0;
        end

        
        function pm_ratio = p_to_m_ratio( obj, ecc )
            pm_ratio = 1;
        end
        
        % The relative value of cortical magnification with respect to ecc=0
        function M = rel_cortical_magnif( obj, ecc )
            e_2 = 3.67; % deg
            %e_2 = obj.par.CM_e_2;
            M = e_2./(ecc+e_2);                    
        end
        
        % Achromatic CSF model
        function S = csf_achrom( obj, freq, area, lum, ach_pars )
            % Internal. Do not call from outside the object.
            % A nested CSF as a function of luminance
            
            N = max( [length(freq) length(area) length(lum)] );
            
            assert( length(freq)==1 || all( size(freq)==[N 1] ) );
            assert( length(area)==1 || all( size(area)==[N 1] ) );
            assert( length(lum)==1 || all( size(lum)==[N 1] ) );
                                    
            S_max = obj.get_lum_dep( ach_pars.S_max, lum );
            f_max = obj.get_lum_dep( ach_pars.f_max, lum );
            bw = obj.get_lum_dep( ach_pars.bw, lum );
            gamma = obj.get_lum_dep( ach_pars.gamma, lum );
            
            S = obj.csf_freq_size( freq, area, S_max, f_max, bw, gamma, ach_pars.Ac_prime );
            
        end
        
        function S = csf_freq_size( obj, freq, area, S_max, f_max, bw, gamma, Ac_prime )
            % log-parabola + Rovamo's stimulus size model
            
            % The stimulus size model from the paper:
            %
            % Rovamo, J., Luntinen, O., & N�s�nen, R. (1993).
            % Modelling the dependence of contrast sensitivity on grating area and spatial frequency.
            % Vision Research, 33(18), 2773�2788.
            %
            % Equation on the page 2784, one after (25)
            
            S_peak = S_max ./ 10.^( (log10(freq) - log10(f_max)).^2./(0.5*2.^bw) );
                                    
            f0 = 0.65;
            
            k = Ac_prime + area.*f0;
            
            S = S_peak .* sqrt( area.^gamma.*freq.^2 ./ (k + area.^gamma.*freq.^2) );
            
        end       

        function pd = get_plot_description( obj )            
            pd = struct();
            pd(1).title = 'Relative cortical magnification';
            pd(1).id = 'rel_cm';
            pd(2).title = 'Sustained and transient response';
            pd(2).id = 'sust_trans';
            pd(3).title = 'Sustained peak sensitivity';
            pd(3).id = 'sust_peak_s';
            pd(4).title = 'Transient peak sensitivity';
            pd(4).id = 'trans_peak_s';
            pd(5).title = 'Sustained peak frequency';
            pd(5).id = 'sust_peak_f';
            %pd(6).title = 'Transient peak frequency';
        end

        function plot_mechanism( obj, plt_id )            
            switch( plt_id )
                case 'rel_cm'
                    clf;
                    html_change_figure_print_size( gcf, 10, 10 );
                    ecc = linspace( 0, 60 );
                    plot( ecc, obj.rel_cortical_magnif( ecc ) );
                    xlabel( 'Eccentricity [deg]' );
                    ylabel( 'Relative cortical magnification' );
                    grid on;
                case 'sust_trans' % sust-trans-response
                    clf;
                    html_change_figure_print_size( gcf, 10, 10 );
                    omega = linspace( 0, 100 );
                    [R_sust, R_trans] = obj.get_sust_trans_resp(omega);
                    hh(1) = plot( omega, R_sust, 'DisplayName', 'Sustained');
                    hold on
                    hh(2) = plot( omega, R_trans, 'DisplayName', 'Transient');
                    hold off
                    xlabel( 'Temp. freq. [Hz]' );
                    ylabel( 'Response' );
                    legend( hh, 'Location', 'Best' );
                    grid on;
                case { 'sust_peak_s', 'trans_peak_s' }
                    clf;
                    html_change_figure_print_size( gcf, 10, 10 );
                    L = logspace( -2, 4 );
                    if strcmp( plt_id, 'sust_peak_s' )
                        S_max = obj.par.ach_sust.S_max;
                    else
                        S_max = obj.par.ach_trans.S_max;
                    end
                    plot( L,  obj.get_lum_dep( S_max, L ) );
                    set_axis_tick_label( 'x', 'luminance', L );
                    set_axis_tick_label( 'y', 'sensitivity', [1 100000] );
                    grid on;
                case 'sust_peak_f'
                    clf;
                    html_change_figure_print_size( gcf, 10, 10 );
                    L = logspace( -2, 4 );
%                     if plt_id == 5
                        f_max = obj.par.ach_sust.f_max;
%                     else
%                         f_max = obj.par.ach_trans.f_max;
%                     end
                    plot( L,  obj.get_lum_dep( f_max, L ) );
                    set_axis_tick_label( 'x', 'luminance', L );
                    set_axis_tick_label( 'y', 'frequency', [0.01 60] );
                    grid on;
                otherwise
                    error( 'Wrong plt_id' );
            end
        end
        
    end
    
    methods( Static )

        function p = get_default_par()

            p = CSF_base.get_dataset_par();

%             p.ach_sust.S_max = [ 2249.56 2.51603 0.230194 10332 0.942421 ];
%             p.ach_sust.f_max = [ 0.794401 76.1359 0.191542 ];
%             p.ach_sust.bw = 1.33448;
%             p.ach_sust.gamma = 0.849947;
%             p.ach_sust.Ac_prime = 1128.63;
%             %p.ach_trans.S_max = [ 993877 4.90261e+143 0.0217163 4.38644e+143 0.0028961 ];
%             p.ach_trans.S_max = [ 1 0 ];
%             p.ach_trans.f_max = [ 0.33223 0.306458 2.34586e-100 ];
%             p.ach_trans.bw = 1.93266;
%             p.ach_trans.gamma = 1.08414;
%             p.ach_trans.Ac_prime = 24.8679;

p.ach_sust.S_max = [ 1708.5 2.36466 0.252301 10849.1 0.879364 ];
p.ach_sust.f_max = [ 0.876388 118.575 0.18265 ];
p.ach_sust.bw = 1.29699;
p.ach_sust.gamma = 0.887729;
p.ach_sust.Ac_prime = 641.114;
p.ach_trans.S_max = [ 0.145321 6.50133 ];
p.ach_trans.f_max = 1.50565e-15;
p.ach_trans.bw = 6.45199;
p.ach_trans.gamma = 1.17238;
p.ach_trans.Ac_prime = 0.138794;
p.sigma_trans = 0.386605;

%p.ach_trans.gamma = 3.09467;
%p.ach_trans.Ac_prime = 4.97325e-09;

        end
        
                
    end
    
end