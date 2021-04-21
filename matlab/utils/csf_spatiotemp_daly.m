function S = csf_spatiotemp_daly( rho, nu )
% rho - spatial frequency in cpd
% nu - temporal frequency in Hz

S = csf_spatiovel_daly( rho, nu./rho );

end