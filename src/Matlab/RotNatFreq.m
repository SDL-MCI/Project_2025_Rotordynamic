function [fn0,fn] = RotNatFreq(xD1,xB2,Ra,R1,R2,Omega)

%% parameters:

L  = 1;                 % length in m    
Ri = 0;                 % inner radius in m
% Ra = 8E-3;              % outer radius in m

% xD1 = 0.4;
% xB2 = 0.7;
xD2 = 0.95;

% R1 = 0.1;
b1 = 0.015;
m1 = R1^2*pi*b1*2700;
% R2 = 0.075;
b2 = 0.015;
m2 = R2^2*pi*b2*2700;

rho = 2700;          % density in kg/m^3
E = 0.7E11;         % Youngs modulus in N/m^2


%% finite-element discretization:

numEL = ceil(xD1/0.04);
xD = linspace(0,xD1,numEL);
xD(end) = [];
numEL = ceil((xB2-xD1)/0.04);
xD = [ xD  linspace(xD1,xB2,numEL)];
xD(end) = [];
numEL = ceil((xD2-xB2)/0.04);
xD = [ xD  linspace(xB2,xD2,numEL)];
xD = [xD L];

numEL = length(xD)-1;
FE.N  = xD;

FE.E  = [ (2:numEL+1)' (1:(numEL))' ];

FE.SC = [ E*ones(numEL,1)  rho*ones(numEL,1)  Ri*ones(numEL,1)  Ra*ones(numEL,1) ];

indBC2 = find(FE.N==xB2);
FE.BC = [ 1  1      % displacement in z-direction
          1  3      % displacement in y-direction
        indBC2  1
        indBC2  3 ];

m = 0.969;
r = 40E-3;
b = 10E-3;
m = r^2*pi*b*8730*2;

indRD1 = find(FE.N==xD1);
indRD2 = find(FE.N==xD2);
FE.RD = [   indRD1   m  0.5*m1*R1^2  0.25*m1*R1^2
            indRD2   m  0.5*m2*R2^2  0.25*m2*R2^2 ];
%% assemble matrices

% number of degrees of freedom
numDOF = length(FE.N)*4;

% initialze system matrices
K = zeros(numDOF,numDOF);
M = zeros(numDOF,numDOF);
G = zeros(numDOF,numDOF);

% --- BEAM ELEMENTS ---
% loop over all beam elements
for i=1:numEL

    % coordinates and length of the element
    x = FE.N(FE.E(i,:));
    l = abs(x(1) - x(2));
    
    % element matrices
    [Me,Ge,Ke] = BeamRDyn(l,FE.SC(i,:));
    
    % local to global degrees of freedom 
    indR = (FE.E(i,1)-1)*4 + 1;
    indL = (FE.E(i,2)-1)*4 + 1;
    ind  = [indR:(indR+3) indL:(indL+3)];
    
    % assembling matrices
    M(ind,ind) = M(ind,ind) + Me;
    G(ind,ind) = G(ind,ind) + Ge;
    K(ind,ind) = K(ind,ind) + Ke;

end

% --- RIGID DISKS ---
numRD = size(FE.RD,1);
for i=1:numRD
    [Me,Ge] = RDisk(FE.RD(i,2:4));

    indN = (FE.RD(i,1)-1)*4 + 1;
    ind  = indN:indN+3;
    M(ind,ind) = M(ind,ind) + Me;
    G(ind,ind) = G(ind,ind) + Ge;
end

% --- APPLY BOUNDARY CONDITIONS ---
constrDOF = (FE.BC(:,1)-1)*4 + FE.BC(:,2);
for i = constrDOF'
    M(i,:) = 0;     M(:,i) = 0;     M(i,i) = 1;
    K(i,:) = 0;     K(:,i) = 0;     K(i,i) = 1;
    G(i,:) = 0;     G(:,i) = 0;     G(i,i) = 0;
end

%% eigenfrequencies at zero speed

[phi0,eigVal0] = eig(K,M);

fn0 = sqrt(diag(eigVal0))/(2*pi);       % Hz
% delete boundary conditions modes
indBC = find(double(fn0>0.99/(2*pi) & fn0<1.01/(2*pi)));
fn0(indBC) = [];

%% eigenfrequencies at certain speed

Mhat = zeros(2*numDOF,2*numDOF);
Khat = zeros(2*numDOF,2*numDOF);

Mhat(1:numDOF,1:numDOF) = Omega*G;
Mhat(1:numDOF,numDOF+1:end) = M;
Mhat(numDOF+1:end,1:numDOF) = eye(numDOF);
Khat(1:numDOF,1:numDOF) = K;
Khat(numDOF+1:end,numDOF+1:end) = -eye(numDOF);

[phi,eigVal] = eig(Khat,-Mhat);

poles = diag(eigVal);
indBC = find(double(abs(imag(poles))<0.2*2*pi));
poles(indBC,:) = [];
phi(:,indBC) = [];

poles = flip(real(poles) + 1i*imag(poles)/(2*pi));
phi = flip(phi,2);
fn  = imag(poles(2:2:end));
phi = phi(:,2:2:end);

[fn,ind] = sort(fn);

end

%% subfunctions

function MAC = compMAC(phi1,phi2)

m = size(phi1,2);
n = size(phi2,2);

for i=1:m
    for j=1:n
        MAC(i,j) = real(phi1(:,i)'*phi2(:,j))^2/((phi1(:,i)'*phi1(:,i))*(phi2(:,j)'*phi2(:,j)));
    end
end

end