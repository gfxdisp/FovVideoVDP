function M = cortical_magnification_dougherty( e )
% Cortical magnification in mm/deg
%
% M = cortical_magnification_dougherty( e )
%
% e - eccentricity in deg
% M - Cortical magnification in mm/deg
%
% From:
% Dougherty, R. F., Koch, V. M., Brewer, A. A., Fischer, B., Modersitzki, J., & Wandell, B. A. (2003). 
% Visual field representations and locations of visual areas v1/2/3 in human visual cortex. 
% Journal of Vision, 3(10), 586–598. https://doi.org/10.1167/3.10.1

% The data from Dougherty 2003
A = 29.2; %mm
e_2 = 3.67; % deg

M = A./(e+e_2);

end