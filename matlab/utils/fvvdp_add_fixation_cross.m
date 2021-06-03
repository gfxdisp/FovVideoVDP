function I_fx = fvvdp_add_fixation_cross( I, pos )

is_color = ndims(I)>2 && size(I,3)==3;

I_fx = I;
fp_size = 8;

if isa( I, 'uint8' ) || isa( I, 'uint16' )
    max_v = single(intmax( class(I) ));
else
    max_v = 1;
end

if is_color
    
    N = size(I,4);

    if size(pos,2)~=2 || (size(pos,1)~=1 && size(pos,1)~=N)
        error( 'Fixation point positions must be a 1x2 or Nx2 matrix' );
    end    
    
    for ff=1:N % For each frame
        p = round(pos(min(size(pos,1),ff),[2 1])+1); % The position of the fixation point
        if any(p<1) || any(p>[size(I,1) size(I,2)])
            continue; % Fixation point outside the frame
        end
        rng = max(1,p(1)-fp_size):min(size(I,1),p(1)+fp_size);
        I_fx(rng,p(2),:,ff)=repmat( cat(3,max_v,max_v,0), [numel(rng) 1 1]);
        rng = max(1,p(2)-fp_size):min(size(I,2),p(2)+fp_size);
        I_fx(p(1),rng,:,ff)=repmat( cat(3,max_v,max_v,0), [1 numel(rng) 1]);                
    end
    
else
    error( 'The function can process only colour video for now' );
end

end
