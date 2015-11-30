% Stratified Sampling
% Input:
%   X: x coordinate matrix
%   Y: y coordinate matrix
%   N1: number of sampling points needed on x
%   N2: number of sampling points needed on y
% Output:
%   xout: sampled points' x coordinates
%   yout: sampled points' y coordinates

function [jitterxout, jitteryout] = stratifiedSample(x,y,N1,N2)

N = size(x,1);
% uniform dividing into boxes
interpx = repmat(x(:,1),1,N1+1)+repmat((x(1,2)-x(1,1))*[0:1/N1:1],N,1);
interpy = repmat(y(:,1),1,N2+1)+repmat((y(1,2)-y(1,1))*[0:1/N2:1],N,1);
% within each smaller box, random generate points
jitterx = repmat(interpx(:,1:N1),1,N2)+(interpx(1,2)-interpx(1,1))*rand(N,N1*N2);
interpytrans = [];
for i = 1:N2
    interpytrans = [interpytrans,repmat(interpy(:,i),1,N2)];
end
jittery = interpytrans +(interpy(1,2)-interpy(1,1))*rand(N,N1*N2);

jitterxout = reshape(jitterx',numel(jitterx),1);
jitteryout = reshape(jittery',numel(jittery),1);

end