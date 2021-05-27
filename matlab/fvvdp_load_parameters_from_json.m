function metric_pars = fvvdp_load_parameters_from_json()
        
param_file_fname = fullfile( fvvdp_data_dir(), 'fvvdp_parameters.json' );
if ~isfile( param_file_fname )
    error( 'JSON file with the metric parameters not found. Missing file: "%s"', param_file_fname );
end
fid = fopen(param_file_fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
metric_pars = jsondecode(str);

end
