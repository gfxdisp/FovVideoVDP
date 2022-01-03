classdef CSF_temp_channels_pm_ratio < CSF_temp_channels_ach
    % Spatio-temporal-eccentricity CSF + PM ratio scaling
    % 
    % The same as CSF_temp_channels but models the change of parvo to magno
    % cells ratio with the eccentricity
    
    properties
        plot_ind;
    end

    methods
        function obj = CSF_temp_channels_pm_ratio()
            obj = obj@CSF_temp_channels_ach();
        end

        function name = short_name( obj )
            % A short name that could be used as a part of a file name
            name = 'ste-csf-temp-ch-pmratio';
        end
        
        function name = full_name( obj )
            name = 'steCSF temp channels pm-ratio';
        end
        
        function pm_ratio = p_to_m_ratio( obj, ecc )
            % Approximated parvo-to-magno ratio from:
            %           
            % Azzopardi, P., K. E. Jones, and A. Cowey. 
            % “Mapping of M and P Projections from the Lateral Geniculate Nucleus to the Striate Cortex in the Macaque Monkey.” 
            % Investigative Ophthalmology and Visual Science 37, no. 3 (1996): 2179–89.
            
%             a1 = 10;
%             b1 = 0.238;
%             pm_base = 1;

            a1 = obj.par.pm_a1;
            b1 = obj.par.pm_b1;
            pm_base = obj.par.pm_base;
            
            pm_ratio = a1*exp(-b1*ecc)+pm_base; % + a2*exp(-b2*ecc);            
            
        end


        function pd = get_plot_description( obj )            
            pd = obj.get_plot_description@CSF_temp_channels_ach();
            ind = length(pd)+1;
            pd(ind).title = 'Ratio of P to M neurons';
            pd(ind).id = 'pm_ratio';
        end

        function plot_mechanism( obj, plt_id )                        
            switch( plt_id )
                case 'pm_ratio'  % PM-ratio
                    clf;
                    html_change_figure_print_size( gcf, 10, 10 );
                    ecc = linspace( 0, 60 );
                    plot( ecc, obj.p_to_m_ratio( ecc ));
                    xlabel( 'Eccentricity [deg]' );
                    ylabel( 'P to M ratio' );
                    grid on;
                otherwise
                    obj.plot_mechanism@CSF_temp_channels_ach( plt_id );
            end
        end
                
    end
    
    methods( Static )

        function p = get_default_par()
%             p = CSF_temp_channels_ach.get_default_par();
% 
%             p.pm_a1 = 10;
%             p.pm_b1 = 0.238;
%             p.pm_base = 1;

p.ach_sust.S_max = [ 237.931 5.8157 0.201395 9121 1.14449 ];
p.ach_sust.f_max = [ 0.793287 25.3648 0.22631 ];
p.ach_sust.bw = 1.46455;
p.ach_sust.gamma = 0.894505;
p.ach_sust.Ac_prime = 158.18;
p.ach_trans.S_max = [ 0.212367 3.92492 ];
p.ach_trans.f_max = 0.000262481;
p.ach_trans.bw = 4.04838;
p.ach_trans.gamma = 1.17018;
p.ach_trans.Ac_prime = 26.1588;
p.sigma_trans = 0.325299;
p.pm_a1 = 11.821;
p.pm_b1 = 0.148939;
p.pm_base = 0.263652;

        end
    end
    

end
