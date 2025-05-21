function [M,G] = RDisk(prop)

m = prop(1);
ThetaP = prop(3);
M = diag([ m  ThetaP  m  ThetaP ]);

ThetaD = prop(2);
G = zeros(4,4);
G(2,4) = -ThetaD;
G(4,2) =  ThetaD;

end