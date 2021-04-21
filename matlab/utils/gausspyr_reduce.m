function y = gausspyr_reduce( x, kernel_a )
% Gaussian pyramid [1] reduction operator. The operator will filder and downsample 
% the input to half of its resolution.
%
% gausspyr_reduce( x )
% gausspyr_reduce( x, kernel_a )
%
% x - input 1-channel image
% kernel_a - parameter of the kernel (0.4 is the default)
%
% [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. 
% IEEE Transactions on Communications, 31(4), 532–540. 
% https://doi.org/10.1109/TCOM.1983.1095851

if ~exist( 'kernel_a', 'var' )    
    kernel_a = 0.4;
end

K=[ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ]; % building kernel

y_a = imfilter( x, K, 'symmetric' ); % filter rows
y_a = y_a(:,1:2:end,:);
y_a = imfilter( y_a, K', 'symmetric' ); % filter columns
y = y_a(1:2:end,:,:);

end
