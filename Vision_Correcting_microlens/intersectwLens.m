% intersectwLens function
% intersection with interference of lens
% Inputs and outputs can be scalars or vectors
%
% Input: 
%   x: The nth sensor pixel matrix x coordinate
%   y: The nth sensor pixel matrix y coordinate
%   u: The microlens matrix x coordinate
%   v: The microlens matrix y coordinate
%   uc: The center of microlens x coordinate
%   vc: The center of microlens y coordinate
%   d1: The distance between starting plane and ending plane
%   d2: The distance between starting plane and intersection plane
%   R: the radius of micro lens
%   check: a binary indicator that determines whether or not to check
%   within R range
% Output:
%   intx: The x coordinate on the intersection plane
%   inty: The y coordinate on the intersection plane

function [intx,inty] = intersectwLens(x,y,u,v,uc,vc,d1,d2)

offVecx = u-uc;
offVecy = v-vc;

rayVecx = uc-x;
rayVecy = vc-y;
dMicro = sqrt(rayVecx.^2+rayVecy.^2);
theta = atan(abs((vc-y)./(uc-x)));

D = abs(dMicro.*(d2/d1));

x_img = sign(d1)*sign(d2)*sign(uc-x).*cos(theta).*D;
y_img = sign(d1)*sign(d2)*sign(vc-y).*sin(theta).*D;

if isempty(x)
    display('Coincidence happens.')
    intx = [];
    inty = [];
    return
end

if(x(1)==uc(1)&&y(1)==vc(1))
    x_img = 0;
    y_img = 0;
end

intx = x_img+x+offVecx;
inty = y_img+y+offVecy;

end