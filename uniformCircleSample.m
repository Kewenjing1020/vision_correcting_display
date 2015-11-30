% Input:
%   x: x coordinates of square boundary
%   y: y coordinates of square boundary
%   N1: sample points along x
%   N2: sample points along y
% Output:
%   uniformx: uniformly sampled x coordinates
%   uniformy: uniformly sampled y coordinates

function [uniformx_out, uniformy_out] = uniformCircleSample(x,y,N_angle,N_point)

R = abs(x(1,1)-x(1,2))/2;
N = size(x,1);
% polar coordinates
angle_step = 2*pi/N_angle;
angle_sample = [0:angle_step:2*pi-angle_step]';
angle_vec = repmat([angle_sample,angle_sample+angle_step],N_point,1);
angle_tol = repmat(angle_vec(:,1),N,1);

diff_temp = angle_vec(1,2)-angle_vec(1,1);
rand_angle = angle_tol+diff_temp.*rand(N_angle*N_point*N,1);
rand_R = R.*sqrt(rand(size(rand_angle)));

uniformx_mat = rand_R.*cos(rand_angle);
uniformy_mat = rand_R.*sin(rand_angle);

uniformx = reshape(uniformx_mat,numel(uniformx_mat),1);
uniformy = reshape(uniformy_mat,numel(uniformy_mat),1);

centerx = sum(x,2)/2;
centery = sum(y,2)/2;

shiftx = reshape(repmat(centerx',N_angle*N_point,1),length(centerx)*N_angle*N_point,1);
shifty = reshape(repmat(centery',N_angle*N_point,1),length(centery)*N_angle*N_point,1);

uniformx_out = uniformx+shiftx;
uniformy_out = uniformy+shifty;
end