classdef fvvdp_display_geometry
    % Use this class to compute the effective resolution of a display in pixels
    % per degree (ppd). The class accounts for the change in the projection
    % when looking at large FOV displays (e.g. VR headsets) at certain
    % eccentricity.
    %
    % The class is also useful for computing the size of a display in meters
    % and visual degrees. Check 'display_size_m' and 'display_size_deg' class
    % properties for that.
    %
    % R = fvvdp_display_geometry( ppd );
    % R = fvvdp_display_geometry( resolution, options );
    %
    % ppd - the fixed value of pixels per degree (if it is already
    % provided)
    %
    % resolution is the 1x2 vector with the pixel resolution of the
    % display: [horizontal_resolutution, vertical_resolutution]
    % The options are name, value pairs. The available options are:
    %
    % distance_m - viewing distance in meters
    % distance_display_heights - viewing distance in the heights of a display
    % fov_horizontal - horizontal field of view of the display in degrees
    % fov_vertical - vertical field of view of the display in degrees
    % fov_diagonal - diagonal field of view of the display in degrees
    % diagonal_size_inches - display diagonal resolution in inches
    % pix_per_deg - pixel per degree assuming that the resolution and
    %               display size is provided. The classwill work out 
    %               the viewing distance.
    %
    % Examples:
    % % HTC Pro
    % % Note that the viewing distance must be specified even though the resolution 
    % % and 'fov_diagonal' are enough to find pix_per_deg.
    % R = fvvdp_display_geometry( [1440 1600], 'distance_m', 3, 'fov_diagonal', 110 );
    % R.get_ppd( [0 10 20 30 40] ); % pix per deg at the given eccentricities
    %
    % % 30" 4K monitor seen from 0.6 meters
    % R = fvvdp_display_geometry( [3840 2160], 'diagonal_size_inches', 30, 'distance_m', 0.6  );
    % R.get_ppd()
    %
    % % 47" SIM2 display seen from 3 display heights
    % R = fvvdp_display_geometry( [1920 1080], 'diagonal_size_inches', 47, 'distance_display_heights', 3 );
    % R.get_ppd()
    %
    % % Find from what viewing distance 30" 4K monitor should be seen to result in 60 pixels per visual degree
    % R = fvvdp_display_geometry( [3840 2160], 'diagonal_size_inches', 30, 'pix_per_deg', 60  );
    % R.distance_m
    %
    % Some information about the effective FOV of VR headsets
    % http://www.sitesinvr.com/viewer/htcvive/index.html
    
    properties
        resolution;
        display_size_m;
        display_size_deg;
        distance_m;
        fixed_ppd = [];
    end
    
    methods
        
        function dr = fvvdp_display_geometry( resolution, varargin )
            
            p = inputParser();
            p.addRequired('resolution',@isnumeric);
            p.addParameter('distance_m',[], @isnumeric);
            p.addParameter('distance_display_heights', [], @isnumeric);
            p.addParameter('fov_horizontal',[], @isnumeric);
            p.addParameter('fov_vertical', [], @isnumeric);
            p.addParameter('fov_diagonal', [], @isnumeric);
            p.addParameter('diagonal_size_inches', [], @isnumeric);
            p.addParameter('pix_per_deg', [], @isnumeric);
            
            p.parse( resolution,varargin{:});
            
            if length(resolution) == 1
                dr.fixed_ppd = resolution;                                               
            else
                
                dr.resolution = resolution;
                
                ar = resolution(1)/resolution(2); % width/height
                
                if ~isempty( p.Results.diagonal_size_inches )
                    height_mm = sqrt( (p.Results.diagonal_size_inches*25.4)^2 / (1+ar^2) );
                    dr.display_size_m = [ar*height_mm height_mm]/1000;
                end
                
                if (~isempty( p.Results.distance_m ) + ~isempty( p.Results.distance_display_heights ) + ~isempty( p.Results.pix_per_deg ))>1
                    error( 'You can pass only one of: ''distance_m'', ''distance_display_heights'', ''pix_per_deg''.' );
                end
                
                if ~isempty( p.Results.distance_m )
                    dr.distance_m = p.Results.distance_m;
                elseif ~isempty( p.Results.distance_display_heights )
                    if isempty( dr.display_size_m )
                        error( 'You need to specify display diagonal size ''diagonal_size_inches'' to specify viwing distance as ''distance_display_heights'' ' );
                    end
                    dr.distance_m = p.Results.distance_display_heights * dr.display_size_m(2);
                elseif ~isempty( p.Results.pix_per_deg )
                    % Find the viewing distance given display size and ppd
                    if isempty(dr.display_size_m)
                        error( 'When ''pix_per_deg'' is passed, you also need to specify ''diagonal_size_inches''. ''pix_per_deg'' works only with flat panel displays.'  );
                    end
                    dr.distance_m = dr.display_size_m(1)/dr.resolution(1) / tand( 1/p.Results.pix_per_deg );
                elseif ~isempty( p.Results.fov_horizontal ) || ~isempty( p.Results.fov_vertical ) || ~isempty( p.Results.fov_diagonal )
                    % Default viewing distance for VR headsets
                    dr.distance_m = 3;
                else
                    error( 'Viewing distance must be specified as ''distance_m'' or ''distance_display_heights'' or ''pix_per_deg'' must be provided.' );
                end
                
                if (~isempty( p.Results.fov_horizontal ) + ~isempty( p.Results.fov_vertical ) + ~isempty( p.Results.fov_diagonal )) >1
                    error( 'You can pass only one of ''fov_horizontal'', ''fov_vertical'', ''fov_diagonal''. The other dimensions are inferred from the resolution assuming that the pixels are square.' );                    
                end
                                
                if ~isempty( p.Results.fov_horizontal )
                    width_m = 2*tand( p.Results.fov_horizontal/2 )*dr.distance_m;
                    dr.display_size_m = [width_m width_m/ar];
                elseif ~isempty( p.Results.fov_vertical )
                    height_m = 2*tand( p.Results.fov_vertical/2 )*dr.distance_m;
                    dr.display_size_m = [height_m*ar height_m];
                elseif ~isempty( p.Results.fov_diagonal )                    
                    % Note that we cannot use Pythagorean theorem on degs -
                    % we must operate on a distance measure
                    % This is incorrect: height_deg = p.Results.fov_diagonal / sqrt( 1+ar^2 );
                    
                    distance_px = sqrt(sum(dr.resolution.^2)) / (2.0 * tand(p.Results.fov_diagonal * 0.5));
                    height_deg = atand( dr.resolution(2)/2 / distance_px )*2;
                    
                    height_m = 2*tand( height_deg/2 )*dr.distance_m;
                    dr.display_size_m = [height_m*ar height_m];
                end
                
                dr.display_size_deg = 2 * atand( dr.display_size_m / (2*dr.distance_m) );
                
            end
            
        end
        
        
        function ppd = get_ppd(dr, eccentricity)
            % Get the number of pixels per degree
            %
            % ppd = R.get_ppd()
            % ppd = R.get_ppd(eccentricity)
            %
            % eccentricity is the viewing angle from the center in degrees. If
            % not specified, the central ppd value (for 0 eccentricity) is
            % returned.
            
            if ~isempty( dr.fixed_ppd )
                ppd = dr.fixed_ppd;
                return;
            end
            
            % pixel size in the centre of the display
            pix_deg = 2*atand( 0.5*dr.display_size_m(1)/dr.resolution(1)/dr.distance_m );
            
            base_ppd = 1/pix_deg;
            
            if ~exist( 'eccentricity', 'var' )
                ppd = base_ppd;
            else
                delta = pix_deg/2;
                tan_delta = tand(delta);
                tan_a = tand( eccentricity );
                
                ppd = base_ppd .* (tand(eccentricity+delta)-tan_a)/tan_delta;
            end
            
        end
        
        function M = get_resolution_magnification(dr, eccentricity)
            % Get the relative magnification of the resolution due to
            % eccentricity.
            %
            % M = R.get_resolution_magnification(eccentricity)
            %
            % eccentricity is the viewing angle from the center in degrees.
            
            if ~isempty( dr.fixed_ppd )
                M = ones(size(eccentricity), 'like', eccentricity );
                return;
            end

            eccentricity = min( eccentricity, 89.9 ); % To avoid singulatities
            
            % pixel size in the centre of the display
            pix_deg = 2*atand( 0.5*dr.display_size_m(1)/dr.resolution(1)/dr.distance_m );
            
            delta = pix_deg/2;
            tan_delta = tand(delta);
            tan_a = tand( eccentricity );
            
            M = (tand(eccentricity+delta)-tan_a)/tan_delta;
        end
        
        function print(dr)
            fprintf( 1, 'Geometric display model:\n' );
            if ~isempty( dr.fixed_ppd )
                fprintf( 1, '  Fixed pixels-per-degree: %g\n', dr.fixed_ppd );
            else
                fprintf( 1, '  Resolution: %d x %d pixels\n', dr.resolution(1), dr.resolution(2) );
                fprintf( 1, '  Display size: %g x %g cm\n', dr.display_size_m(1)*100, dr.display_size_m(2)*100  );
                fprintf( 1, '  Display size: %g x %g deg\n', dr.display_size_deg(1), dr.display_size_deg(2)  );
                fprintf( 1, '  Viewing distance: %g m\n', dr.distance_m );
                fprintf( 1, '  Pixels-per-degree (center): %g\n', dr.get_ppd() );
            end
        end
        
    end
    
    methods( Static )
        
        function dg = create_from_json( dm_struct )
            
            if isfield( dm_struct, 'viewing_distance_meters' )
                viewing_distance_meters = dm_struct.viewing_distance_meters;
            elseif isfield( dm_struct, 'viewing_distance_inches' )
                viewing_distance_meters = 0.0254*dm_struct.viewing_distance_inches;
            else
                error( 'Display model entry must specify viewing distance' );
            end
            
            if isfield( dm_struct, 'diagonal_size_inches' )            
                dg = fvvdp_display_geometry( dm_struct.resolution, 'diagonal_size_inches', dm_struct.diagonal_size_inches, 'distance_m', viewing_distance_meters );
            elseif isfield( dm_struct, 'fov_diagonal' )
                dg = fvvdp_display_geometry( dm_struct.resolution, 'fov_diagonal', dm_struct.fov_diagonal, 'distance_m', viewing_distance_meters );
            else
                error( 'Cannot parse display geometry from JSON' );
            end
            
        end
        
    end
    
    
end

