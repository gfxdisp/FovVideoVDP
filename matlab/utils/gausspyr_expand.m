function y = gausspyr_expand( x, sz, kernel_a )
% Gaussian pyramid [1] expand operator. The operator will upsample the input to
% twice its resolution.
%
% y = gausspyr_expand( x )
% y = gausspyr_expand( x, sz )
% y = gausspyr_expand( x, sz, kernel_a )
%
% x - input 1-channel image
% sz - size [height width] of the resulting image y
% kernel_a - parameter of the kernel (0.4 is the default)
%
% [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. 
% IEEE Transactions on Communications, 31(4), 532–540. 
% https://doi.org/10.1109/TCOM.1983.1095851

if ~exist( 'sz', 'var' )    
    sz = [size(x,1) size(x,2)]*2;
end

if ~exist( 'kernel_a', 'var' )    
    kernel_a = 0.4;
end

ch_no = size(x,3); % How many colour channels

K=2*[ 0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2 ]; % building kernel

% Note that we need a custom padding of 2 pixels that would avoid putting
% two non-0 rows/columns next to each other

y_a = zeros( [size(x,1) sz(2)+4 ch_no], 'like', x ); % we add 2 pix padding on both sides
y_a(:,3:2:(end-2),:) = x;
y_a(:,1,:) = x(:,1,:); % padding
y_a(:,(end-1):end,:) = y_a(:,(end-3):(end-2),:); % padding
y_a = imfilter( y_a, K, 0 ); % filter rows

y = zeros( [sz(1)+4 sz(2) ch_no], 'like', x );
y(3:2:(end-2),:,:) = y_a(:,3:(end-2),:);
y(1,:,:) = y_a(1,3:(end-2),:); % padding
y((end-1):end,:,:) = y((end-3):(end-2),:,:); % padding
y = imfilter( y, K', 0 ); % filter columns
y = y(3:(end-2),:,:);

end
