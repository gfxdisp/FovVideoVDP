function L = srgb2lin( p )
% Convert pixel values to linear RGB using sRGB non-linearity
% 
% p = lin2srgb( L )
%
% p - pixel values
% L - relative linear RGB (or luminance), normalized to the range 0-1
%
% (c) 2012 Rafal Mantiuk

L = zeros(size(p), 'like', p);

t = 0.04045;
a = 0.055;

L(p<=t) = p(p<=t)/12.92;
L(p>t) = ((p(p>t)+a)/(1+a)).^2.4;

end