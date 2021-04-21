function LMS = xyz2lms2006( XYZ )
% Transform from CIE 2006 LMS cone responses to CIE 1931 XYZ trichormatic colour values on an LED LCD display. 
%
% LMS = xyz2lms2006( XYZ )
%
% XYZ can be an image (h x w x 3) or a colour vectors (n x 3).
%
% Note that because CIE XYZ 1931 and CIE 2006 LMS are based on different
% colour matching functions, this transformation depends on the colour
% spectra of a display. This transformation was derived for the spectra of LED LCD. 
%
% The CIE 2006 LMS cone responses can be found at http://www.cvrl.org/.

M_xyz_lms2006 = [ 
   0.187596268556126   0.585168649077728  -0.026384263306304;
  -0.133397430663221   0.405505777260049   0.034502127690364;
   0.000244379021663  -0.000542995890619   0.019406849066323 ];
  
LMS = cm_colorspace_transform( XYZ, M_xyz_lms2006 );

end

