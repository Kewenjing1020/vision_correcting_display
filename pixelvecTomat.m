% function pixelReorder
% Input:
%   pixelVec: a N by 1 vector where N is the total pixel numbers at the
%   sensro side
% Output:
%   pixelMat: the reordered image matrix

% P: Microlens number
% N: Sensor pixel number
function pixelMat = pixelvecTomat(pixelVec, P, N)

pixelMat = zeros(P*N);

for i = 1:N
    for j = 1:N
        for k = 1:P
            pixelMat((i-1)*P+k,(j-1)*P+1:j*P) = ...
                pixelVec((i-1)*N*P*P+(j-1)*P*P+(k-1)*P+...
                1:(i-1)*N*P*P+(j-1)*P*P+(k)*P);
        end
    end
end

return
