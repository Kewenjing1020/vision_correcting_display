% intersection function
% Input:
%   x: The nth sensor pixel matrix x coordinate
%   y: The nth sensor pixel matrix y coordinate
%   u: The microlens matrix x coordinate
%   v: The microlens matrix y coordinate
%   d1: The distance between starting plane and ending plane
%   d2: The distance between starting plane and intersection plane

function [intx,inty] = intersection(x,y,u,v,d1,d2)
rayVecx = u-x;
rayVecy = v-y;
dMicro = sqrt(rayVecx.^2+rayVecy.^2);
theta = atan(abs((v-y)./(u-x)));

D = abs(dMicro.*(d2/d1));

x_img = sign(d1)*sign(d2)*sign(u-x).*cos(theta).*D;
y_img = sign(d1)*sign(d2)*sign(v-y).*sin(theta).*D;

if isempty(x)
    display('Coincidence happens.')
    intx = [];
    inty = [];
    return
end

if(x(1)==u(1)&&y(1)==v(1))
    x_img = 0;
    y_img = 0;
end

intx = x_img+x;
inty = y_img+y;
end