function map = hdrvdp_visualize( p1, varargin )
% HDRVDP_VISUALIZE produces color or grayscale visualization of the metric
% predictions
%
% map = HDRVDP_VISUALIZE( P )
% map = HDRVDP_VISUALIZE( P, context_image )
% map = HDRVDP_VISUALIZE( P, context_image, target )
% map = HDRVDP_VISUALIZE( P, context_image, target, colormap )
% map = HDRVDP_VISUALIZE( 'pmap', P, options )
% map = HDRVDP_VISUALIZE( 'diff', P, test, reference, options )
% map = HDRVDP_VISUALIZE( 'civdm', civdm )
% map = HDRVDP_VISUALIZE( 'civdm', civdm, context_image )
%
% Creates a colorful visualization from the HDR-VDP-2 results. Depending on
% whether 'diff' or 'pmap' type is selected, the function visualizes a
% different aspect of the prediction. The older syntax with the first
% parameter 'P' is equivalent to 'pmap'.
%
% 'pmap' produces the probability of detection map. Such a map shows where 
%        and how likely a difference will be noticed. However, It does not 
%        show what this differece is. 
%
%        The required argument 'P' is the probability of detection map. P 
%         must be within the range 0-1
%
% 'diff' shows the contrast-normalized per-pixel difference weighted by the
%        probability of detection. The resulting images do NOT show
%        probabilities. However, they better correspond to the perceived
%        differences and thus are easier to interpret.
%     
%        The required argument 'P' is the probability of detection map. P 
%         must be within the range 0-1
%
%        The two required argumants are the test and references images, the
%        same as passed to the hdrvdp.
%
%  'civdm' produces the visualization for the Contrast Invariant Visual
%        Difference Metric (a.k.a. Dynamic Range Independent Metric). 
%        You need to pass 'civdm' structure generated by the HDR-VDP-3 with
%        the 'task' set to 'civdm'. The color coding is as follows:
%          red - contrast reversal
%          green - contrast loss
%          blue - contrast amplification
%        Refer to the paper: 
%          T. O. Aydin, R. Mantiuk, K. Myszkowski, and H.-P. Seidel, 
%          “Dynamic range independent image quality assessment,” 
%          ACM Trans. Graph. (Proc. SIGGRAPH), vol. 27, no. 3, p. 69, 2008.
%
% 'options' a cell array with { 'name', value } pairs. Refer to the examples
%        below. Available options:
%
%    'context_image' (default: []) - display context image on top of a
%        colour map. This is usually a 'test' image passed to the hdrvdp.
%
%
%    'target' (default 'screen') - specifies desired output device and can 
%        be one of:
%
%        'screen' - (default) - the map be shown on a color screen. 
%                 The map will contain good reproduction of the context
%                 image 'img'. 
%  
%        'print' - the map can be printed on a gray-scale printer, so color
%                  information will be lost. If this target is selected,
%                  the color map will contain luma differences in addition
%                  to color differences. To ensure that the context image
%                  does not interfere with errors, only low-contrast and
%                  high frequency content of the image will be preserved. 
%
%     'colormap' (default 'trichromatic') parameter can be one of:
%
%        'trichromatic' - Errors are represented as multiple colors: blue,
%                  cyan, green, yellow and red, which correspond to P equal
%                  0, 0.25, 0.5, 0.75 and 1.  
%
%        'dichromatic' - more appropriate if observes may be color
%                  deficient. The hue changes from cyan (0.0) to gray (0.5)
%                  and then yellow (1.0). The look of images for
%                  color-deficient observers can be tested at: 
%             http://www.colblindor.com/coblis-color-blindness-simulator/
%
%        'monochromatic' - use only grayscale. Makes sense only with
%                  target='print' or when no context image is specified
% 
%
% Tone-mapping is applied to the context image to reduce the dynamic range
% so that highly saturated colors can be used also in bright image regions.
% Tone-mapping will enhance contrast if it is lower than the available
% contrast. This behavior is useful for images that have contrast that is
% near detection threshold (e.g. ModelFest stimuli).
%
% The function returns a gamma-corrected sRGB image.
%
% Example:
%
% reference = logspace( log10(0.1), log10(1000), 512 )' * ones( 1, 512 );
% test = reference .* (1 + (rand( 512, 512 )-0.5)*0.02);
% res = hdrvdp( test, reference, 'luminance', 30 );
%
% imshow( hdrvdp_visualize( 'pmap', res.P_map, { 'context_image', test } ) )
%
% Legend for the color-scales can be found in the color_scales
% directory.
%
% Copyright (c) 2010-2019, Rafal Mantiuk

% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
%  * Redistributions of source code must retain the above copyright notice, 
%    this list of conditions and the following disclaimer.
%  * Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the  documentation
%    and/or other materials provided with the distribution.  
%  * Neither the name of the HDR-VDP nor the names of its contributors may be
%    used to endorse or promote products derived from this software without 
%    specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY 
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if( ischar( p1 ) ) % new syntax
    switch( p1 )
        case 'diff'
            [P, test_img, reference_img, options] = set_params( varargin, 'required', 'required', 'required', {} );
            opt = parse_options( options );
            P = norm_diff_img( test_img, reference_img ) .* P;
        case 'pmap'
            [P, options] = set_params( varargin, 'required', {} );
            opt = parse_options( options );            
        case 'civdm'
            [civdm, context_image] = set_params( varargin, 'required', {} );
            map = generate_civdm_map( civdm, context_image );
            return;
        otherwise
            error( 'Urecognizaed visuzalization type' );
    end
    
else % old syntax
    P = p1;
    opt = struct();    
    [opt.context_image, opt.target, opt.colormap] = set_params( varargin, [], 'screen', 'trichromatic' );
end

if( isempty(opt.context_image) )
    tmo_img = ones(size(P))*0.5;
elseif( strcmp( opt.target, 'print' ) )   
    l = log_luminance( opt.context_image );
    hp_img = (l - blur_gaussian( l, 2 ) + mean(l(:)));
    tmo_img = vis_tonemap(hp_img, 0.1) + 0.5;
elseif( strcmp( opt.target, 'screen' ) )
    tmo_img = vis_tonemap( log_luminance(opt.context_image), 0.6 );
else
    error( 'Unknown target: %s', opt.target );
end

P(P<0) = 0;
P(P>1) = 1;


    if( strcmp( opt.colormap, 'trichromatic' ) || strcmp( opt.colormap, 'print' ) )
        
        color_map = [0.2  0.2  1.0;
            0.2  1.0  1.0;
            0.2  1.0  0.2;
            1.0  1.0  0.2;
            1.0  0.2  0.2];
        
        color_map_in = [0 0.25 0.5 0.75 1];
        
    elseif( strcmp( opt.colormap, 'dichromatic' ) )
        
        color_map = [0.2  1.0  1.0;
            1.0  1.0  1.0
            1.0  1.0  0.2];
        
        color_map_in = [0 0.5 1];

    elseif( strcmp( opt.colormap, 'monochromatic' ) )
        
        color_map = [1.0  1.0  1.0;
            1.0  1.0  1.0];
        
        color_map_in = [0 1];        
        
    else
        error( 'Unknown colormap: %s', opt.colormap );
    end
    

    if( strcmp( opt.target, 'screen' ) )
        color_map_l = color_map * [0.2126 0.7152 0.0722]'; %sum(color_map,2);
        color_map_ch = color_map ./ repmat( color_map_l, [1 3] );
    else
        if( strcmp( opt.colormap, 'monochromatic' ) )
            color_map_l = (color_map * [0.2126 0.7152 0.0722]') ./ color_map_in';
        else
            % luminance map start at 0.3, so that colors are visible
            color_map_l = (color_map * [0.2126 0.7152 0.0722]') ./ (color_map_in'*0.8+0.2);
        end
        color_map_ch = color_map ./ repmat( color_map_l, [1 3] );
    end
    
    
    %The line belows display pixel values
    %round(min( color_map_ch*255, 255 ))
    
    
    map = zeros( size(P,1), size(P,2), 3, size(P,3) );
    map(:,:,1,:) = interp1( color_map_in, color_map_ch(:,1), P );
    map(:,:,2,:) = interp1( color_map_in, color_map_ch(:,2), P );
    map(:,:,3,:) = interp1( color_map_in, color_map_ch(:,3), P );
    
    %map = repmat( tmo_img, [1 1 3] );
    if size(tmo_img,3)==1 % image
        map = map .* repmat( tmo_img, [1 1 3] );
    else % video
        map = map .* repmat( reshape( tmo_img, [size(tmo_img,1) size(tmo_img,2) 1 size(tmo_img,3)] ), [1 1 3] );
    end

end

function l = log_luminance( X )

if( size(X,3) == 3 )
    Y = X(:,:,1) * 0.212656 + X(:,:,2) * 0.715158 + X(:,:,3) * 0.072186;
else
    Y = X;
end

Y(Y<=0) = min(Y(Y>0));
l = log(Y);

end

function tmo_img = vis_tonemap( b, dr )   
    % b is an image given as log(luminance)
    
    t = 3;
    
    b_min = min(b(:));
    b_max = max(b(:));
    
    if b_max-b_min < dr % No tone-mapping needed
        tmo_img = (b-b_min)/(b_max-b_min+1e-3)*dr + (1-dr)/2;
        return;
    end
    
    b_scale = linspace( b_min, b_max, 1024 );
    b_p = hist( b(:), b_scale );
    b_p = b_p / sum( b_p(:) );
    
    sum_b_p = sum( b_p.^(1/t) );
    dy = b_p.^(1/t) / sum_b_p;
    
    v = cumsum( dy )*dr + (1-dr)/2;
    
    tmo_img = interp1( b_scale, v, b );
end

function Y = blur_gaussian( X, sigma )
ksize2 = round(sigma*3);
gauss_1d = exp( -(-ksize2:ksize2).^2/(2*sigma^2) );
gauss_1d = gauss_1d/sum(gauss_1d);

Y = conv2( X, gauss_1d, 'same' );
Y = conv2( Y, gauss_1d', 'same' );

end

function varargout = set_params( vals, varargin )


for kk=1:length(varargin)
    
    if( length(vals) >= kk )
        varargout(kk) = vals(kk);
    else
        if( ischar( varargin{kk} ) && strcmp( 'required', varargin{kk} ) )
            error( 'Required parameter missing' );
        else
            varargout(kk) = varargin(kk);
        end
    end
    
end

end


function I = norm_diff_img( test, reference )
% Computer contrast-normalized difference image

D = get_luminance(test)-get_luminance(reference);

sigma = 5;
window = fspecial('gaussian', round(sigma*4), sigma);
img_mu = filter2(window, D, 'same');
sigma_sq = max(0, filter2(window, D.^2, 'same' ) - img_mu.^2);
v = ( sigma_sq ).^(1/2);

I = min(D./(v+1),1);

end


function Y = get_luminance( img )
% Return 2D matrix of luminance values for 3D matrix with an RGB image

%dims = sum(nnz( size(img)>1 ));
dims = find(size(img)>1,1,'last');

if( dims == 3 )
    Y = img(:,:,1) * 0.212656 + img(:,:,2) * 0.715158 + img(:,:,3) * 0.072186;
elseif( dims == 1 || dims == 2 )
    Y = img;
else
    error( 'get_luminance: wrong matrix dimension' );
end

end

function opt = parse_options( options )

opt = struct();

valid_opts = { 'colormap', 'trichromatic', 'target', 'screen', 'context_image', [] };

for kk=1:2:length(options)
    
    opt_found = false;
    for ll=1:2:length(valid_opts)
        if( strcmp( options{kk}, valid_opts{ll} ) )
            opt_found = true;
            valid_opts{ll+1} = 'done';
            break;
        end
    end
    if( ~opt_found )
        error( 'Unrecognized option: %s', options{kk} );
    end
    opt.(options{kk}) = options{kk+1};
end

%Insert default options
for kk=1:2:length(valid_opts)
    if( ~ischar( valid_opts{kk+1} ) || ~strcmp( valid_opts{kk+1}, 'done' ) )
        opt.(valid_opts{kk}) = valid_opts{kk+1};
    end
end

end

function map = generate_civdm_map( civdm, context_image )

%sz = size( civdm.loss );

map = cat( 3, civdm.rev, civdm.loss, civdm.ampl );

% Keep only the strongest distortion at each pixel
max_v = max(map,[],3);
map(map < repmat(max_v, [1 1 3])) = 0;

if ~isempty( context_image )
    tmo_img = vis_tonemap( log_luminance(context_image), 0.6 );
    alpha = 0.5;
    map = (1-alpha)*map + alpha*repmat(tmo_img.^(1/2.2),[1 1 3]);
end

end


