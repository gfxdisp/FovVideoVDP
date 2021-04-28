function path = fvvdp_data_dir()
% Return the directory with the configuration JSON files for FovVideoVDP. 
% Those files are shared by both Matlab and Python version of the metric.  

pathstr = fileparts(mfilename( 'fullpath' ));
path = fullfile( pathstr, '..', 'fvvdp_data' );

end