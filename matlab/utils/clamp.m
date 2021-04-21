function Y = clamp( X, min, max )
% Y = clamp( X, min, max )
% 
% Restrict values of 'X' to be within the range from 'min' to 'max'.

  Y = X;
  Y(X<min) = min;
  Y(X>max) = max;
end
