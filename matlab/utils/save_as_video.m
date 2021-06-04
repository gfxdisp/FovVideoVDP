function save_as_video( vid, fname, fps, profile )

if ~exist( 'fps', 'var' )
    fps = 30;
end

% doc writeVideo for the full list of profiles
if ~exist( 'profile', 'var' )
    profile = 'MPEG-4';
end


v = VideoWriter( fname, profile );
v.FrameRate = fps;
if strcmp( profile, 'MPEG-4' )
    % Supported only by selected profiles
    v.Quality = 95;
end
open(v);

if length(size(vid))==4 && size(vid,3)==3 % Colour video
    writeVideo(v,vid);
else
    writeVideo(v,clamp(reshape(vid, [size(vid,1) size(vid,2) 1 size(vid,3)]), 0, 1) );
end

close(v);

end