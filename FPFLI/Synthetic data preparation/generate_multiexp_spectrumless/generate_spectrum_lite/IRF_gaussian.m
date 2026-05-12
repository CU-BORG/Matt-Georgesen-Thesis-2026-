function I = IRF_gaussian(t0, h, FWHM)
% t0: start bin
% h: bin width
% FWHM: full width half maximum
% @author: mg 

t=1:256;
sig0 = FWHM/2.3548/h;
I=exp(-(t-t0).^2/(2*sig0^2));

end