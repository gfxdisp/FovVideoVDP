function ver = fvvdp_version( )
metric_par = fvvdp_load_parameters_from_json();
ver = metric_par.version;
end