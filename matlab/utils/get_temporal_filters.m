function [F, omega] = get_temporal_filters( frames_per_s, f_len, sigma_s, beta )
% f_len - how many taps should temporal filters have
% F{1} - sustained
% F{2} - transient

t = linspace( 0, f_len/frames_per_s, f_len )'; 

% Temporal filters based on Smith's paper
if ~exist( 'sigma_s', 'var' )
    sigma_s = 0.5;
end
if ~exist( 'beta', 'var' )
    beta = 0.06;
end

epsilon=1e-4;
F = cell(2,1);

if 0
F{1} = exp( -(log(t+epsilon)-log(beta)).^2/(2*sigma_s^2) ); % Sustained 
%F{1} = F{1} / mean(F{1}); % So that the modulation at DC is 1
F{1} = F{1} / sum(F{1}); % So that the energy of the contrast remains the same

%F{2} = 0.061*cat( 1, diff( F{1} )/(t(2)-t(1)), 0 ); % Transient
F{2} = cat( 1, diff( F{1} )/(t(2)-t(1)), 0 ); % Transient

F_test = F{1} .* (log(t+epsilon)-log(beta))./(sigma_s^2*(t+epsilon));

% Point in time at which the transient channel reaches the peak response
t_m = beta*exp(- (sigma_s*(sigma_s^2 + 4)^(1/2))/2 - sigma_s^2/2) - epsilon; 
t = linspace( 0, f_len/frames_per_s, f_len );

% The normalization constant
n_C = sum(sin( 2*pi*t/t_m/4 ).*F{2}');

F{2} = F{2}/n_C;
elseif 1

    % Redone using the correct normalization constant for the transient this
    % time
    
F{1} = exp( -(log(t+epsilon)-log(beta)).^2/(2*sigma_s^2) ); % Sustained 
F{1} = F{1} / sum(F{1}); % So that the energy of the contrast remains the same

k2 = 0.062170507756932;
% This one seems to be slightly more accurate at low sampling rates
F{2} = k2*cat( 1, diff( F{1} )/(t(2)-t(1)), 0 ); % Transient

%F{2} = -k2 * F{1} .* (log(t+epsilon)-log(beta))./(sigma_s^2*(t+epsilon));

else
    % From Winkler 2005
    % Something seems wrong with this formula
    
    F{1} = exp( -log((t/1000+epsilon)-0.160)/0.2 );

    F{1} = F{1} / sum(F{1}); % So that the energy of the contrast remains the same

    k2 = 0.12;
    F{2} = k2*cat( 1, diff( F{1}, 2 )/(t(2)-t(1)), 0 ); % Transient
end

%omega = [1 5];
omega = [0 5];

end