function [uniformx, uniformy] = uniformSample(x,y,N1,N2)

N = size(x,1);
% uniform dividing into boxes
uniformx = repmat(x(:,1),1,N1)+repmat((x(1,2)-x(1,1))*[0:1/(N1-1):1],N,1);
uniformy = repmat(y(:,1),1,N2)+repmat((y(1,2)-y(1,1))*[0:1/(N2-1):1],N,1);

end