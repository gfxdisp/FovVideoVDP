function [Q_JOD, diff_map, Q] = fvvdp(test_video, reference_video, varargin)
% This is a high-level interface to Foveated Video Visual Differece
% Predictor or FovVideoVDP - a full reference video/image quality metric
% accounting for the physical display specification (size, viewing
% distance), foveated viewing, and temporal aspects of vision. 
%
% Q_JOD = fvvdp(test_image, reference_image)
% [Q_JOD, diff_map, Q] = fvvdp(test_image, reference_image)
% [...] = fvvdp(test_video, reference_video, 'frames_per_second', 30)
% [...] = fvvdp(test_video_file, reference_video_file)
% [...] = fvvdp(..., 'display_name', 'htc_vive_pro' )
% [...] = fvvdp(..., 'display_name', '?' )
% [...] = fvvdp(..., 'display_photometry', disp_photo )
% [...] = fvvdp(..., 'display_geometry', disp_geo )
% [...] = fvvdp(..., 'color_space', 'rec2020' )
% [...] = fvvdp(..., 'foveated', true|[false] )
% [Q_JOD, diff_map] = fvvdp(..., 'heatmap', 'threshold' )
% [...] = fvvdp(..., 'options', { 'fixation_point', [100 100] } )
% [...] = fvvdp(..., 'quiet', true )
%
% test_image and reference_image must be either a path to a video file or 
% a tensor of the size:
% [height x width x 1] for grayscale image 
% [height x width x 3] matrix for color image
% [height x width x 1 x frames] for grayscale video
% [height x width x 3 x frames] matrix for color video
%
% If image/video is supplied as a tensor, you must specify 'frames_per_second',
% parameter when passing video.
%
% If file name is provided, the video frames are read by Matlab's
% VideoReader class.
%
% Q_JOD - is the image quality on the 10-level scale, with the highest
%         quality at 10. 
%
% 'display_name' specifies one of the displays listed in
% display_models/display_models.json. The entries from the JSON file
% specify both display geometry and its phtometry. The default is
% 'standard_4k', which is a 30-inch 4K SDR display, seen from 60 cm. 
% Pass '?' to list available displays. Pass '--list' to get the detailed
% list with all the parameters. 
%
% 'display_photometry' - the parameter is an object of the class
% fvvdp_display_photometry (or its subclass) which specifies how to decode
% pixel values and convert those into luminance emitted from the display. 
%
% 'display_geometry' - the parameter is an object of the class
% fvvdp_display_geometry (or its subclass) which specifies display size and
% the viewing distance. 
%
% If only 'display_photometry' or 'display_geometry' is specified, the
% other part of display specification is taken from the default display
% specification ('standard_4k');
%
% 'color_space' specifies the color space in which color data is stored. The
% available color spaces are listed in display_models/color_spaces.json.
% The default is 'rec709' (widely used for SDR content, aka sRGB).
%
% 'foveated', false (default)/true - whether to assume foveated (true) or
% regular viewing (false). Regular viewing means that each portion of the 
% image is attended. If 'foveated' is set to true, the observer is looking at
% the central point in the image or the position given by the 'fixation_point'
% option (see below). 
%
% 'heatmap' if heatmap is specified, 'diff_map' will contain color-coded
% visualization of the difference map. The possible values are 
%   'threshold' - use blue-green-yellow-red color map to denote values from
%                 0 to 1 JOD
%   'supra-threshold' - use cyan-yellow map to denote JND values from 0 to
%                 3 JOD.
%   The images with the labelled color scale can be found in 'color_scales' 
%   folder.
%
% 'options' allow to pass extra options to the metric as a cell array of 
%    'name', value pairs. The most relevant options are:
%
%    'fixation_point', [x, y] - the pixel coordinates of the fixation
%          point. The left-top corner has the coordinates [0,0]. If none is
%          specified, the metric assumes that the fixation point is in the
%          centre of the image. To simulate a moving gaze point, you can pass 
%          a matrix of the size [N 2] where N is the number of frames. 
%
%    'frame_padding', type - the metric requires at least 250ms of video to
%          for temporal processing. Because no previous frames exist in the
%          first 250ms of video, the metric must pad those first frames.
%          This options specifies the type of padding to use.
%          'replicate' - replicate the first frame
%          'circular' - tile the video in the front, so that the last frame 
%                       is used for frame 0.
%
%    'use_gpu', true/false - set to false if you do not have CUDA-capable
%          graphics card. The etric will be much slower. 
%
% This function should be sufficient for 90% of use cases, but you may need
% to use fvvdp_core if you need access to the low-level features.
%
% The details of the metric are described in the paper: 
% FovVideoVDP: A visible difference predictor for wide field-of-view video

p = inputParser();

valid_input = @(in) isnumeric(in) || ischar(in);
p.addRequired('test_video',valid_input);
p.addRequired('reference_video',valid_input);
p.addParameter('frames_per_second', 0, @(x)validateattributes(x,{'numeric'},{'nonempty','nonnegative'}) );
p.addParameter('display_name', 'standard_4k', @ischar);
p.addParameter('display_photometry', []);
p.addParameter('display_geometry', [], @(x) isa(x,'fvvdp_display_geometry') );
p.addParameter('foveated', false, @islogical );
p.addParameter('color_space', 'rec709', @ischar);
p.addParameter('options', {}, @iscell);
p.addParameter('heatmap', [], @(x) ismember(x, { 'threshold', 'supra-threshold' }) );
p.addParameter('quiet', false, @islogical );

p.parse(test_video, reference_video, varargin{:});

if ~(~isempty( p.Results.display_photometry ) && ~isempty( p.Results.display_geometry ))    
    % Only when both models are provided, we do not need to read the JSON
    % file
    disp_list = fvvdp_video_source.load_json( 'display_models.json' );
    
    if strcmp( p.Results.display_name, '?' ) || strcmp( p.Results.display_name, '--list' )
        fn = fieldnames(disp_list);
        fprintf( 'Available displays:\n' );
        for kk=1:length(fn)
            fprintf( '''%s'' - %s\n', fn{kk}, disp_list.(fn{kk}).name );
            if strcmp( p.Results.display_name, '--list' )
                dm_struct = disp_list.(fn{kk});
                display_geom = fvvdp_display_geometry.create_from_json( dm_struct );
                display_ph_model = fvvdp_display_photometry.create_from_json( dm_struct );
                display_geom.print();
                display_ph_model.print();
            end
        end
        return
    end
    
    if ~isfield( disp_list, p.Results.display_name )
        error( 'Unknown display model "%s". Check fvvdp_data/display_models.json for available display models.', p.Results.display_name );
    end
    dm_struct = disp_list.(p.Results.display_name);
end

if ~isempty( p.Results.display_photometry )
    if ischar( p.Results.display_photometry ) && strcmp( p.Results.display_photometry, 'absolute' )
        display_ph_model = fvvdp_display_photo_absolute();
    else
        display_ph_model = p.Results.display_photometry;
    end
else                            
    display_ph_model = fvvdp_display_photometry.create_from_json( dm_struct );
end

if ischar( test_video ) && ischar( reference_video ) % If privided with file names
    vs = fvvdp_video_source_sdr_file( test_video, reference_video, display_ph_model, p.Results.color_space );
else % If provided with tensors
    vs = fvvdp_video_source_sdr( test_video, reference_video, p.Results.frames_per_second, display_ph_model, p.Results.color_space );
end

if ~isempty( p.Results.display_geometry )
    display_geom = p.Results.display_geometry;
else
    display_geom = fvvdp_display_geometry.create_from_json( dm_struct );
end

if ~p.Results.quiet
    if exist( 'dm_struct', 'var' ) && isfield( dm_struct, 'name' )
        fprintf( 1, 'Display name: %s\n', dm_struct.name );
    end
    display_geom.print();
    display_ph_model.print();
    vid_sz = vs.get_video_size();
    if length(vid_sz)==2 || vid_sz(3)==1 
        content_type = 'Image';
    else
        content_type = 'Video';
        fprintf( 1, 'Frames per second: %g\n', vs.get_frames_per_second() );
    end    
    fprintf( 1, '%s resolution: [%d %d] pix\n', content_type, vid_sz(2), vid_sz(1) );    
    if isempty(display_geom.fixed_ppd) && (vid_sz(1)<display_geom.resolution(2) || vid_sz(2)<display_geom.resolution(1))
        fprintf( 1, '  Content is smaller then the display resolution. The metric assumes that the image is shown at the native display resolution in the central portion of the screen.\n' );
    end
    if p.Results.foveated
        fprintf( 1, 'Foveated mode.\n' )
        fv_mode = 'foveated';
    else
        fprintf( 1, 'Non-foveated mode (default).\n' )
        fv_mode = 'non-foveated';
    end
    
    metric_par = fvvdp_load_parameters_from_json();
    fprintf( 1, 'When reporting metric results, please include the following information:\n' );
    
    if startsWith(p.Results.display_name, 'standard_') && isempty( p.Results.display_photometry ) && isempty( p.Results.display_geometry )
        % append this if are using one of the standard displays
        standard_str = strcat( ', (', p.Results.display_name, ')' );
    else
        standard_str = '';
    end
    fprintf( 1, '"FovVideoVDP v%.01f, %.4g [pix/deg], Lpeak=%.5g, Lblack=%.4g [cd/m^2], %s%s"\n', metric_par.version, display_geom.get_ppd(), display_ph_model.get_peak_luminance(), display_ph_model.get_black_level(), fv_mode, standard_str );
    
end
options = p.Results.options;

if nargout > 1 % the metric runs faster if no diff_map is needed
    options = cat( 2, options, 'do_diff_map', true );    
end    

if p.Results.foveated
    options = cat( 2, options, 'do_foveated', true );    
end

res = fvvdp_core( vs, display_geom, options );

Q_JOD = res.Q_JOD;

if nargout > 1
    if strcmp( p.Results.heatmap, 'threshold' )        
        diff_map = hdrvdp_visualize( 'pmap', res.diff_map, { 'context_image', vs.get_reference_video(false), 'colormap', 'trichromatic' } );
    elseif strcmp( p.Results.heatmap, 'supra-threshold' )        
        diff_map = hdrvdp_visualize( 'pmap', res.diff_map/3, { 'context_image', vs.get_reference_video(false), 'colormap', 'dichromatic' } );
    else
        diff_map = res.diff_map;
    end                
end

if nargout > 2
    Q = res.Q;
end

end
