function G = weber2log( W )
% Convert Weber contrast 
%
% W = (B-A)/A
%
% to log contrast
%
% G = log10( B/A );
%

G = log10(W+1);

end