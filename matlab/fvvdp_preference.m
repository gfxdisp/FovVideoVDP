function [p_pref, p_A_better] = fvvdp_preference( JOD_A, JOD_B )
% The function converts the difference between two JOD scores into
% a percentage increase in preference (p_pref) and the probability of
% selecting A over B (p_A_better). 
%
% If JOD_B>JOD_A, the function will return negative values. 
%
% Both values are returned as percentages (between -100 and 100). If no
% output variables are assigned, the function will print the
% interpretation of the quality difference. 
%
% The function can accept vectors and matrices of the same size as input. 
% 
% Please refer to "Predicted quality scores" on the README.md.

sigma_cdf = 1.4826; % The standard deviation for the JOD/JND units (1 JOD = 0.75 p_A_better)

p_A_better = normcdf( JOD_A-JOD_B, 0, sigma_cdf ) * 100;
p_pref = (p_A_better*2-100);

if nargout==0 && numel(JOD_A)==1
    if p_pref>0
        cond = 'AB';        
    else
        cond = 'BA';
    end
    fprintf( 'Condition %s shows %.3g%% increase in preference over condition %s.\n', cond(1), abs(p_pref), cond(2) );
    fprintf( '%.3g%% of the population will select condition A over condition B.\n', p_A_better );
end

end