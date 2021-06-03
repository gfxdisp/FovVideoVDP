function I_fx = fvvdp_add_fixation_cross( V, fixation_point )
% Add a yellow fixation cross in the video I at the fixation positions
% fixation_point. The fixation_point should be in the same format as 
% fixation_point option of fvvdp_core: 
% in pixel coordinates (x,y), where x=0..width-1 and y=0..height-1
% fixation_point must be either [1 2] or [N 2] matrix where N is the 
% number of frames. 

is_color = ndims(V)>2 && size(V,3)==3;

I_fx = V;
fp_size = 8;

if isa( V, 'uint8' ) || isa( V, 'uint16' )
    max_v = single(intmax( class(V) ));
else
    max_v = 1;
end

if is_color
    
    N = size(V,4);

    if size(fixation_point,2)~=2 || (size(fixation_point,1)~=1 && size(fixation_point,1)~=N)
        error( 'Fixation point positions must be a 1x2 or Nx2 matrix' );
    end    
    
    for ff=1:N % For each frame
        p = round(fixation_point(min(size(fixation_point,1),ff),[2 1])+1); % The position of the fixation point
        if any(p<1) || any(p>[size(V,1) size(V,2)])
            continue; % Fixation point outside the frame
        end
        rng = max(1,p(1)-fp_size):min(size(V,1),p(1)+fp_size);
        I_fx(rng,p(2),:,ff)=repmat( cat(3,max_v,max_v,0), [numel(rng) 1 1]);
        rng = max(1,p(2)-fp_size):min(size(V,2),p(2)+fp_size);
        I_fx(p(1),rng,:,ff)=repmat( cat(3,max_v,max_v,0), [1 numel(rng) 1]);                
    end
    
else
    error( 'The function can process only colour video for now' );
end

end
