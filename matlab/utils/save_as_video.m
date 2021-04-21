function save_as_video( vid, fname, fps )

if ~exist( 'fps', 'var' )
    fps = 30;
end

v = VideoWriter( fname, 'MPEG-4' );
v.FrameRate = fps;
v.Quality = 95;
open(v);

if length(size(vid))==4 && size(vid,3)==3 % Colour video
    writeVideo(v,vid);
else
    writeVideo(v,clamp(reshape(vid, [size(vid,1) size(vid,2) 1 size(vid,3)]), 0, 1) );
end

close(v);

end