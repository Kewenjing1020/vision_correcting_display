% visualization

X1 = jitterx;
Y1 = jittery;
Z1 = zeros(length(jitterx),1);

X2 = inputx_pinhole;
Y2 = inputy_pinhole;
Z2 = FOV_micro*ones(length(X2),1);

figure()
for i = 1:250
    X = [X1(i),X2(i)];
    Y = [Y1(i),Y2(i)];
    Z = [Z1(i),Z2(i)];
    
    plot3(X,Y,Z,'Color','r','LineWidth',1);
    hold on
end

%%
dislen = 625;
figure()
plot3(jitterx(1:dislen),jittery(1:dislen),zeros(dislen,1),'.');
hold on
plot3(inputx_pinhole(1:dislen),inputy_pinhole(1:dislen),5.51*ones(dislen,1),'.');
axis equal

%%
dislen = 625*3;
figure()
for i = 1:dislen
    plot3([jitterx(i),inputx_pinhole(i)],[jittery(i),inputy_pinhole(i)],...
        [0,5.51]);
    hold on
    plot3([jitterx(i),inttempx(i)],[jittery(i),inttempy(i)],...
        [0,20]);
    hold on
    plot3(obj(i,2),obj(i,3),20,'o','markerfacecolor',[obj(i,1),obj(i,1),obj(i,1)],...
        'markeredgecolor',[obj(i,1),obj(i,1),obj(i,1)])

end
axis equal;

%%
numPoint = N1*Num_sensorPixel*Num_pinhole;
jitterxmat = reshape(jitterx,numPoint,numPoint);
sliceLCDx = jitterxmat(numPoint/2,:);
sliceLCDy = zeros(length(sliceLCDx),1);
pinholexmat = reshape(inputx_pinhole,numPoint,numPoint);
slicePinholex = pinholexmat(numPoint/2,:);
slicePinholey = 5.51*ones(length(slicePinholex),1);
angle = slicePinholex - sliceLCDx;
figure()
plot(sliceLCDx,angle,'.');

figure()
plot(sliceLCDx,sliceLCDy,'.')
