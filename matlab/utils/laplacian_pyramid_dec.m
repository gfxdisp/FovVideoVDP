function [lpyr, gpyr, gpyr_exp] = laplacian_pyramid_dec(img, levels, kernel_a)
% A decimated laplcian pyramid (every levels is 1/2 of the resolution). 
% The functions implements the laplacian pyramid according to a paper:
% The Laplacian Pyramid as Compact Image Code
%      by Peter J.Burt and Edward H.Adelson
%
% Usage:
%   laplacian_pyramid_dec(img,levels,kernel_a)
%
% img - image in grey scale - matrix <height x width double>
% levels - height of pyramid , cannot be bigger then log_2(min(width,height)),
%          with levels=-1, the hight is equal to floor(log_2(min(width,height)))
% kernel_a - it is used for generating kernel for diffusion,
%       a method for that is given in the paper
%
% lpyr - cell array of matrices with the Laplacian pyramd, each higher
%        level halves the resolution
% gpyr - cell array of matrices with the Gaussian pyramd
% gpyr_exp - a cell array with expanded levels of the Gaussian pyramid
%       (used to compute the Laplacian pyramid)
%
% If no 
%
% It can be used also in such ways:
%   laplacian_pyramid_dec(img) - levels set to be largest
%                           kernel_a set to be equal to 0.4
%   laplacian_pyramid_dec(img,levels) - kernel_a set to be equal to 0.4

if ~exist( 'kernel_a', 'var' )
    kernel_a = 0.4;
end

if ~exist( 'levels', 'var' )
    levels = -1;
end

do_gpyr_exp = (nargout == 3); % Whether to generate the expanded gpyr

g_pyramid=gaussian_pyramid_dec(img,levels,kernel_a);

height = numel(g_pyramid);
if height == 0
    lpyr={};
    return;
end

lpyr = cell(height,1);
if do_gpyr_exp
    gpyr_exp = cell(height,1);
end
for i=1:height-1
    gpyr_lp1 = gausspyr_expand( g_pyramid{i+1}, [size(g_pyramid{i},1) size(g_pyramid{i},2)], kernel_a );    
    if do_gpyr_exp
        gpyr_exp{i+1} = gpyr_lp1;
    end
    lpyr{i} = g_pyramid{i} - gpyr_lp1;    
end

lpyr{height} = g_pyramid{height};
if nargout > 1
    gpyr = g_pyramid;
end
