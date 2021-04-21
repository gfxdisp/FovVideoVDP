function out = cm_colorspace_transform( in, M )
% Transform an image or color vector (Nx3) to another color space using matrix M
% 
% out = cm_colorspace_transform( in, M )
%
% "in" could be an image [width x height x 3] or [rows x 3] matrix of colour
%     vectors
% M is a colour transformation matrix so that
%     dest = M * src (where src is a column vector)
%

if any( size(M) ~= [3 3] )
    error( 'Colour transformation must be a 3x3 matrix' );
end

if length(size(in)) == 2 
    out = in * M';
else    
    out = reshape( (M * reshape( in, [size(in,1)*size(in,2) 3] )')', ...
        [size(in,1) size(in,2) 3] );
end

end