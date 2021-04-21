function S = csf_spatiovel_daly( rho, vel )
% Spatio-velocity contrast sensitivity function from the paper of Laird et
% al. 
% 
% S = csf_spatiovel_daly( rho, vel )
% 
% rho - spatial frequency in cycles per degree [cpd]
% vel - retinal velocity in deg / sec
%
% The original model is undefined for vel=0. The rationale could be that
% the sensitivity is 0 for a stationary image on the retina (the image
% disappers without any eye movements). To prevent NaNs, the model has
% been modified so that the minimum velocity is min_vel=0.1 [deg/sec]. 
% 0.1 [deg/sec] approximately corresponds to the minimum velocity of
% the eye during a fixation.

min_vel = 0.1; % The minimum velocity

s1 = 6.1;
s2 = 7.3;
p1 = 45.9;

if( 1 )
   % Fit based on the paper:
   % [1] J. Laird, M. Rosen, J. Pelz, E. Montag, and S. Daly, “Spatio-velocity CSF as a function of retinal velocity using unstabilized stimuli,�? 2006, vol. 6057, p. 605705.
   
   c0 = 0.6329;
   c1 = 0.8404;
   c2 = 0.7986;

   vel_clamped = max( vel, min_vel ); % Added to avoid NaN for vel=0
   
   k = s1 + s2 * abs(log10(c2*vel_clamped/3)).^3;
   rho_max = p1 ./ (c2*vel_clamped+2);
   
   S = k * (c0 * c1 * c2) .* vel_clamped .* (c1*2*pi*rho).^2 .* exp( - (c1*4*pi*rho)./(rho_max) );
   %S = k .* vel .* (2*pi*rho).^2 .* exp( -4*pi* rho .* (vel+2)./p1);

else
    % This one is based on:
    % [1] S. J. Daly, “Engineering observations from spatiovelocity and spatiotemporal visual models,�? 1998, vol. 3299, no. January, pp. 180–191.
    % but corrected for the formula accoding to:
    % [1] K. Myszkowski, P. Rokita, and T. Tawara, “Perception-based fast rendering and antialiasing of walkthrough sequences,�? IEEE Trans. Vis. Comput. Graph., vol. 6, no. 4, pp. 360–379, 2000.
    % These are older papers so it is better to use the calibrated model
    % above.
    
    c0 = 1.14;
    c1 = 0.67;
    c2 = 1.7;
    
    k = s1 + s2 * abs(log10(c2*vel/3)).^3;
    %S = k * (c0 * c2) .* vel .* (2*pi*c1*rho).^2 .* exp( - (c1*4*pi*rho .*(c2*vel+2)./p1) );

    S = k .* vel .* (2*pi*rho).^2 .* exp( -4*pi* rho .* (vel+2)./p1);
    
end



end