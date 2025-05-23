function  plotWhirl(FE,phi,Omega,pole)

numN = length(FE.N);

alpha = linspace(0,4*pi,100);

for i=1:numN
    
    numDOF = (i-1)*4;
	w = phi(numDOF+1);
    v = phi(numDOF+3);
    
    phiW = angle(w);
    phiV = angle(v);
   
    whirlW(:,i) = real(w.*exp(1i*alpha));
    whirlV(:,i) = real(v.*exp(1i*alpha));
    whirlX(:,i) = FE.N(i)*ones(length(alpha),1);
end


if phiV-phiW<0 && phiV-phiW>-pi
    mode = 'BW';
else
    mode = 'FW';
end

close all
figure

line([ FE.N(FE.E(:,1)) ; FE.N(FE.E(:,2)) ],zeros(2,numN-1),zeros(2,numN-1),'Color',[ 0 0 0 ])
line(FE.N,zeros(2,numN),zeros(2,numN),'Color',[ 0 0 0 ],'LineStyle','none','Marker','o','MarkerSize',3,'MarkerFaceColor',[1 1 0.0667])

line(whirlX,whirlV,whirlW,'LineStyle',':','Color',[ 0.502  0.502  0.502 ],'LineWidth',0.5)


j = 1;
h = line(FE.N,whirlV(j,:),whirlW(j,:),'Color',[0  0.4471  0.7412],'LineWidth',2);
hP = line(FE.N,whirlV(j,:),whirlW(j,:),'LineStyle','none','Color',[0  0.4471  0.7412],'Marker','o','MarkerSize',4,'MarkerFaceColor',[0  0.4471  0.7412],'MarkerEdgeColor',[0  0.4471  0.7412]);

view(36,24)
xlabel 'x / mm'
ylabel 'v / mm'
zlabel 'z / mm'
grid on
box on

YLim = get(gca,'YLim');
ZLim = get(gca,'ZLim');
maxD = max(max(YLim),max(ZLim));
set(gca,'YLim',[-maxD maxD],'ZLim',[-maxD maxD]);

set(gca, 'Zdir', 'reverse')
set(gca, 'Ydir', 'reverse')

maxW = max(max(abs(whirlW)));
maxV = max(max(abs(whirlV)));

maxU = max(maxW,maxV);
L = abs(max(FE.N) - min(FE.N));

scal = 0.3*L/maxU;

set(gca,'DataAspectRatioMode','manual','DataAspectRatio',[scal 1 1]);

title([mode,': \Omega = ',num2str(Omega*60/(2*pi),'%4.0f'),' rpm   f_n = ',num2str(pole,'%4.2f'),' Hz'])

numDt = length(alpha);
for i=1:numDt
    h.YData = whirlV(i,:);
    h.ZData = whirlW(i,:);
    hP.YData = whirlV(i,:);
    hP.ZData = whirlW(i,:);
    drawnow;
end

