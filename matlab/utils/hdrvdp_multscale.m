classdef hdrvdp_multscale
    
    methods( Abstract )
        
        ms = decompose( ms, I );
        I = reconstruct( ms );
        
        B = get_band( ms, band, o );
        ms = set_band( ms, band, o, B );
        
        bc = band_count( ms );
        oc = orient_count( ms, band );

        % Get band frequences in cyc/deg
        bf = get_freqs( ms );
        
    end
    
end