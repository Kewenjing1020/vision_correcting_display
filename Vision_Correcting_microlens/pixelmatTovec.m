
% P: Microlens number
% N: Sensor pixel number
function pixelVec = pixelmatTovec(pixelMat, P, N)

pixelVec = zeros(size(pixelMat,1)*size(pixelMat,2),1);

for i = 1:N
    for j = 1:N
        for k = 1:P
            pixelVec((i-1)*P*P*N+(j-1)*P*P+(k-1)*P+1:(i-1)*P*P*N+(j-1)*P*P+k*P)=...
                pixelMat((i-1)*P+k,(j-1)*P+1:j*P);
        end
    end
end

return