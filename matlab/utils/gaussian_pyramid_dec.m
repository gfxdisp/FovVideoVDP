function res = gaussian_pyramid_dec(img,levels,kernel_a)
% Function which builds decimated gaussian pyramid according to the paper:
% The Laplacian Pyramid as Compact Image Code
%      by Peter J.Burt and Edward H.Adelson
%
% Usage:
%   gaussian_pyramid(img,levels,kernel_a)
%
% img - image in grey scale - matrix <height x width double>
% levels - height of pyramid , cannot be bigger then log_2(min(width,height)),
%          with levels=-1, the hight is equal to floor(log_2(min(width,height)))
% kernel_a - it is used for generating kernel for diffusion, 
%       a method for that is given in the paper
%
% res - cell array of matrix <height x width double>
%
% It can be used also in such ways:
%   gaussian_pyramid(img) - levels set to be largest
%                           kernel_a set to be equal to 0.4
%   gaussian_pyramid(img,levels) - kernel_a set to be equal to 0.4

    if nargin == 0
        error( 'Incorrect number of parameters!' );
    end
        
    default_levels = floor(log2(min(size(img,1),size(img,2))));

    switch nargin
        case 1
            levels = -1;
            kernel_a = 0.4; % value for the paper
        case 2
            kernel_a = 0.4; %value from the paper
    end
            
    if levels == -1
        levels = default_levels;
    end
    
    if levels > default_levels
        error( 'parameter "levels" to large!' );
    end
            
    res = cell(levels,1);
    res{1}=img;
    if levels==1
        return
    end
    for i=2:levels
        res{i} = gausspyr_reduce( res{i-1}, kernel_a );
    end
    if size(res{levels},1) == 0
        res={};
    end

end
