classdef fvvdp_content_mapping 
    
    properties
%         img_sz;
    end
    
    methods
%         function cm = fvvdp_content_mapping( img_sz )
%             cm.img_sz = img_sz;
%         end
        
        function ecc = get_eccentricity_map( cm, img_sz, fixation_point )
            
            [phi, theta] = meshgrid( linspace(0,2*pi,img_sz(2)), linspace(0,pi,img_sz(1)) );
            
            theta_fix = (fixation_point(2)-1)/(img_sz(1)-1) * pi;
            phi_fix = (fixation_point(1)-1)/(img_sz(2)-1) * 2*pi;
            
            xyz_fix = cm.spherical2euclidean( [theta_fix phi_fix] );
            
            xyz = cm.spherical2euclidean( cat( 2, theta(:), phi(:) ) );
            
            ecc = acosd(clamp(dot( xyz, repmat(xyz_fix,[size(xyz,1) 1]), 2 ),-1,1));                                    
            ecc = reshape( ecc, img_sz );
            
        end
        
        function [xyz] = spherical2euclidean( cm, sph )
            
            xyz = [ sin(sph(:,1)).*cos(sph(:,2)) sin(sph(:,1)).*sin(sph(:,2)) cos(sph(:,1)) ];
            
        end
        
    end
    
end