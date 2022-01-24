classdef CSF_temp_channels_ecdrop < CSF_base
    % Spatio-temporal-eccentricity CSF
    % Rely on achromatic (single colour mechanism) CSF
    %
    % Decompose into sustained and trasient channels
    % Separate CSF for both
    % Pyramid of Visibility drop of sensitivity with eccentricity
    % Rovamo's spatial integration model
    
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
        
        function obj = CSF_temp_channels_ecdrop( )                        
            obj.par = obj.get_default_par();                                   
        end

        function name = short_name( obj )
            % A short name that could be used as a part of a file name
            name = 'ste-csf-temp-ecdrop';
        end
        
        function name = full_name( obj )
            name = 'steCSF temp linear drop with eccentricity';
        end

        function S = sensitivity( obj, csf_pars )
            
            csf_pars = obj.test_complete_params(csf_pars, { 'luminance', 'ge_sigma' } );

            ecc = csf_pars.eccentricity;
            sigma = csf_pars.ge_sigma;
            rho = csf_pars.s_frequency;
            omega = csf_pars.t_frequency;
            lum = csf_pars.luminance;
                        
            [R_sust, R_trans] = get_sust_trans_resp(obj, omega);
           
            A = pi*(sigma).^2; % Stimulus area
                        
            pm_ratio = obj.p_to_m_ratio(ecc);
            
            S_sust = obj.csf_achrom( rho, A, lum, ecc, obj.par.ach_sust );
             
%             ach_trans = obj.par.ach_trans;
%             ach_trans.S_max = [1 ach_trans.S_max_sh];
%             S_trans = obj.csf_achrom( rho_cm, A_cm, lum, ach_trans );

            %S_trans = zeros(size(omega)); 
            S_trans = obj.csf_achrom( rho, A, lum, ecc, obj.par.ach_trans );
            

            S_aux = obj.aux_sensitivity( csf_pars );
            if obj.ps_beta ~= 1 
                beta = obj.ps_beta;
                S = ( (R_sust.*S_sust .* sqrt(pm_ratio)).^beta + (R_trans.*S_trans .* sqrt(1./pm_ratio)).^beta + S_aux.^beta).^(1/beta);
            else
                S = R_sust.*S_sust .* sqrt(pm_ratio) + R_trans.*S_trans .* sqrt(1./pm_ratio) + S_aux;
            end
            
            % The drop of sensitivity with the eccentricity (the window of
            % visibiliy model)
            alpha = min(1, abs(csf_pars.vis_field-180)/90 );
            ecc_drop = alpha .* obj.par.ecc_drop + (1-alpha) .* obj.par.ecc_drop_nasal;            
            ecc_drop_f = alpha .* obj.par.ecc_drop_f + (1-alpha) .* obj.par.ecc_drop_f_nasal;            
            a = ecc_drop + rho.*ecc_drop_f;
            S = S .* 10.^(-a.*ecc);

        end


        % Get the sustained and transient temporal response function
        function [R_sust, R_trans] = get_sust_trans_resp(obj, omega)
            sigma_sust = obj.par.sigma_sust;
            beta_sust = 1.3314;
%             sigma_trans = 0.2429;
%             beta_trans = 0.1898;            
            %omega_0 = obj.par.omega_0; %5;
            omega_0 = 5;

             beta_trans = 0.1898;            
             sigma_trans = obj.par.sigma_trans; % 2.5690;

            R_sust = exp( -omega.^beta_sust / (sigma_sust) );
            R_trans = exp( -abs(omega.^beta_trans-omega_0^beta_trans).^2 / (sigma_trans) );            
        end
        
        function S = aux_sensitivity( obj, csf_pars )
            S = 0;
        end

        
        function pm_ratio = p_to_m_ratio( obj, ecc )
            pm_ratio = 1;
        end
        
        % The relative value of cortical magnification with respect to ecc=0
        function M = rel_cortical_magnif( obj, ecc, vis_field )
            %e_2 = 3.67; % deg
            %e_2 = obj.par.cm_e2;

            cm_e2 = obj.par.cm_e2;
            cm_e2_nasal = obj.par.cm_e2_nasal;
%            cm_e2 = 2.71423*8;
%            cm_e2_nasal = 2.74799*8;
            
            alpha = min(1, abs(vis_field-180)/90 );
            e_2 = alpha .* cm_e2 + (1-alpha) .* cm_e2_nasal;
            

            M = e_2./(ecc+e_2);                    
        end
        
        % Achromatic CSF model
        function S = csf_achrom( obj, freq, area, lum, ecc, ach_pars )
            % Internal. Do not call from outside the object.
            % A nested CSF as a function of luminance
            
            N = max( [length(freq) length(area) length(lum)] );
            
            assert( length(freq)==1 || all( size(freq)==[N 1] ) );
            assert( length(area)==1 || all( size(area)==[N 1] ) );
            assert( length(lum)==1 || all( size(lum)==[N 1] ) );
                                    
            S_max = obj.get_lum_dep( ach_pars.S_max, lum );
            f_max = obj.get_lum_dep( ach_pars.f_max, lum );
            %gamma = obj.get_lum_dep( ach_pars.gamma, lum );           

            bw = ach_pars.bw;
            %bw = max( 0.01, ach_pars.bw - ach_pars.ecc_bw_drop*ecc);
            a = ach_pars.a;

            % Truncated log-parabola 
            S_LP = 10.^( -(log10(freq) - log10(f_max)).^2./(0.5*2.^bw) );
            ss = (freq<f_max) & (S_LP < (1-a));
            S_LP(ss) = 1-a;            

            S_peak = S_max .* S_LP;
            

            % The stimulus size model from the paper:
            %
            % Rovamo, J., Luntinen, O., & N�s�nen, R. (1993).
            % Modelling the dependence of contrast sensitivity on grating area and spatial frequency.
            % Vision Research, 33(18), 2773�2788.
            %
            % Equation on the page 2784, one after (25)
%             f0 = 0.65;            
%             k = ach_pars.Ac_prime + area.*f0;
            
            if isfield( ach_pars, 'f0' )
                f0 = ach_pars.f0;
            else
                f0 = 0.65;
            end
            if isfield( ach_pars, 'A0' )
                A0 = ach_pars.A0;
            else
                A0 = 270;
            end

            Ac = A0./(1+(freq/f0).^2);
            
            S = S_peak .* sqrt( Ac ./ (1+Ac./area)).*(freq.^1);
            %S = S_peak .* sqrt( Ac ./ (1+Ac./area.^gamma)).*(freq.^1);
            %S = S_peak .* sqrt( area.^gamma.*freq.^2 ./ (k + area.^gamma.*freq.^2) );
            
        end
        
        function pd = get_plot_description( obj )            
            pd = struct();
            pp = 1;
            pd(pp).title = 'Sustained and transient response';
            pd(pp).id = 'sust_trans';
            pp = pp+1;            
            pd(pp).title = 'Sustained peak sensitivity';
            pd(pp).id = 'sust_peak_s';
            pp = pp+1;
            pd(pp).title = 'Transient peak sensitivity';
            pd(pp).id = 'trans_peak_s';
            pp = pp+1;
            pd(pp).title = 'Sustained peak frequency';
            pd(pp).id = 'sust_peak_f';
            pp = pp+1;
        end

        function plot_mechanism( obj, plt_id )            
            switch( plt_id )
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
                    hold on
                    L_dvr = logspace( -1, 1 );
                    plot( L_dvr, sqrt(L_dvr)*100, '--k' );

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

% Fitted on 2022/01/24
p.ach_sust.S_max = [ 67.3403 29.4202 0.196765 8.12573e-07 7.03859e+09 ];
p.ach_sust.f_max = [ 1.47635 54.2739 0.246014 ];
p.ach_sust.bw = 1.18863;
p.ach_sust.a = 0.0828388;
p.ach_trans.S_max = [ 0.538269 63.3312 ];
p.ach_trans.f_max = 0.0156452;
p.ach_trans.bw = 2.81858;
p.ach_trans.a = 0.000273289;
p.sigma_trans = 0.133094;
p.sigma_sust = 13.8756;
p.ecc_drop = 0.0268966;
p.ecc_drop_nasal = 0.0134587;
p.ecc_drop_f = 0.0200277;
p.ecc_drop_f_nasal = 0.0170489;

        end
        
                
    end
    
end