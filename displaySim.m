% Ray Tracer of 2D image
% Created on Oct.5th
% by Xuaner Zhang
% Last revised: Nov.21st

% Input: system parameter as a struct that contains
%   N: multiview pixel number
%   D: distance between microlens and sensor pixel
%   P: view-dependent sub-pixel number
%   A N by N microlens cell that specifies
%       position of each microlens
%       an array of pixel struct under one microlens
%   the position of the display (center pixel location)
%   the position of the object/image
%   the position of the observer/image plane

% Output: A 2D image
%   Each pixel on the image plane has the intensity of the average of the
%   sensor pixels emitting on sit

%% System setup
clc
close all
warning off
%%%%%%%%%%%%%%%%%%%%%% Image Specification %%%%%%%%%%%%%%%%%%%%%%
image = imread('test_image/testimg_rose_square_color.jpg');
% lena_color = imread('lena512color.tiff');
img_display = im2double(image);
if(length(size(image))==3)
    [img_sizex,img_sizey,~] = size(img_display);
    imgVecTotal = [reshape(img_display(:,:,1)',[img_sizex*img_sizey,1]),...
        reshape(img_display(:,:,2)',[img_sizex*img_sizey,1]),...
        reshape(img_display(:,:,3)',[img_sizex*img_sizey,1])];
else
    [img_sizex,img_sizey] = size(img_display);
    imgVecTotal = reshape(img_display',[img_sizex*img_sizey,1]);
end

img_x = 0;
img_y = 0;
pinhole_x = 0;
pinhole_y = 0;
FOV_micro = 5.51;
pinhole_z = FOV_micro;
display_img_res = 0.078;
Num_pinhole = 128;
Num_sensorPixel = 5; % directional resolution

pinhole_block = Num_pinhole*Num_pinhole;
pinhole_phy_size = Num_sensorPixel*Num_pinhole*display_img_res;
physSizex_pinhole = pinhole_phy_size;
physSizey_pinhole = pinhole_phy_size;
pinhole_pitch = physSizex_pinhole/Num_pinhole;
pinhole_R = pinhole_pitch*15/78/2;
pinhole = zeros(pinhole_block,2);

[pinhole_X,pinhole_Y] = meshgrid(1:Num_pinhole,1:Num_pinhole);
pinholex_vec = (pinhole_X-1-Num_pinhole/2)*pinhole_pitch+pinhole_pitch/2+pinhole_x;
pinholey_vec = (-pinhole_Y+1+Num_pinhole/2)*pinhole_pitch-pinhole_pitch/2+pinhole_y;
pinhole(:,1:2) = [reshape(pinholex_vec',numel(pinholex_vec),1),...
    reshape(pinholey_vec',numel(pinholey_vec),1)];
%%%%%%%%%%%%%%%%%%%%%% LCD Plane %%%%%%%%%%%%%%%%%%%%%%
physSizex_sensorPixel = physSizex_pinhole;
physSizey_sensorPixel = physSizey_pinhole;
sensorPixel_pitch = pinhole_pitch/Num_sensorPixel;
sensorPixel_R = sensorPixel_pitch;
bond_sensorPixel = [min(pinhole(:,1)),max(pinhole(:,1));...
    min(pinhole(:,2)),max(pinhole(:,2))];

LCD_x = 0;
LCD_y = 0;
LCD_z = 0;

sensorpixel_block = Num_sensorPixel*Num_sensorPixel;
sensorPixel = zeros(sensorpixel_block,3);

[LCD_X,LCD_Y] = meshgrid(1:Num_pinhole*Num_sensorPixel,1:Num_pinhole*Num_sensorPixel);
LCDx_vec = (LCD_X-1-Num_pinhole*Num_sensorPixel/2)*sensorPixel_pitch+sensorPixel_pitch/2+LCD_x;
LCDy_vec = (-LCD_Y+1+Num_pinhole*Num_sensorPixel/2)*sensorPixel_pitch-sensorPixel_pitch/2+LCD_y;
LCD_xy_temp = [reshape(LCDx_vec',numel(LCDx_vec),1),...
    reshape(LCDy_vec',numel(LCDy_vec),1)];
LCDX_vec = reshape(LCD_X,numel(LCD_X),1);
LCDY_vec = reshape(LCD_Y,numel(LCD_Y),1);
LCD_blockx = ceil(LCDX_vec/Num_sensorPixel)-1;
LCD_blocky = ceil(LCDY_vec/Num_sensorPixel)-1;
mod_LCDx = mod(LCDX_vec,Num_sensorPixel);
mod_LCDx(mod_LCDx==0)=Num_sensorPixel;
mod_LCDy = mod(LCDY_vec,Num_sensorPixel);
mod_LCDy(mod_LCDy==0)=Num_sensorPixel;
LCD_index = [LCD_blockx*sensorpixel_block*Num_pinhole+...
    LCD_blocky*sensorpixel_block+(mod_LCDx-1)*Num_sensorPixel+mod_LCDy]';
sensorPixel(LCD_index,2:3) = LCD_xy_temp;
sensorPixel(:,1) = reshape(repmat([1:pinhole_block],sensorpixel_block,1),...
    sensorpixel_block*pinhole_block,1);


%% LCD Simulator
% object depth
obj_z_vec = [20];

corner_LCD = sensorPixel(1,2);
corner_pinhole = pinhole(1,1)+pinhole_R;

for mm = 1:length(obj_z_vec)
    obj_z = obj_z_vec(mm);
    
    % set the object size
    corner_obj = corner_LCD+abs(corner_pinhole-corner_LCD)*...
        (pinhole_z-LCD_z)/(obj_z-pinhole_z);
    
    physSizex_img = 2*abs(corner_obj);
    physSizey_img = physSizex_img;
    
    obj_res = physSizex_img/img_sizex;
    obj = zeros(img_sizex*img_sizey,3);
    [obj_X,obj_Y] = meshgrid(1:img_sizex,1:img_sizey);
    objx_vec = (obj_X-1-img_sizex/2)*obj_res+obj_res/2+img_x;
    objy_vec = (-obj_Y+1+img_sizey/2)*obj_res-obj_res/2+img_y;
    obj(:,2:3) = [reshape(objx_vec',numel(objx_vec),1),...
        reshape(objy_vec',numel(objx_vec),1)];
    
    if(obj_z<0)
        obj_z = pinhole_z-obj_z+pinhole_z;
    end
    
    obj_bond = [min(obj(:,2)),max(obj(:,2));...
        min(obj(:,3)),max(obj(:,3))];
    fprintf('Defining object resolution to be %f ppi.\n',25.4/display_img_res);
    
    % imgPlane
    %   1 col: intensity at the pixel
    %   2 col: x position in world coordinate
    %   3 col: y position in world coordinate
    
    displaySensor = zeros(Num_pinhole*Num_sensorPixel,...
        Num_pinhole*Num_sensorPixel,size(imgVecTotal,2));
    %     for channel = 1:size(imgVecTotal,2)
    %         imgVec = imgVecTotal(:,channel);
    %         obj(:,1) = imgVec;
    
    % LCD Simulator
    fprintf('Begin LCD Simulator.\n');
    K_img = 1; % k nearest neighbors (filtering parameter)
    % assume gray scale image
    sensorColor = zeros(size(sensorPixel,1),size(sensorPixel,2)+1);
    sensorColor(:,1:size(sensorPixel,2)) = sensorPixel;
    
    %%%%%%%%%%%%%%%% stratified sampling at LCD plane %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % within each sensor pixel, sample 25 points randomly using stratified
    % jittering algorithm
    N1 = 5;
    N2 = 5;
    xinput_sensor = [sensorPixel(:,2)-sensorPixel_R,sensorPixel(:,2)+sensorPixel_R];
    yinput_sensor = [sensorPixel(:,3)+sensorPixel_R,sensorPixel(:,3)-sensorPixel_R];
    
    fprintf('Jittered sampling 25 points on LCD plane.\n');
    [jitterx, jittery] = stratifiedSample(xinput_sensor,yinput_sensor,N1,N2);
    
    %%%%%%%%%%%%%%%% uniform sampling at micro-lens plane %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % within each micro lens, sample #directional resolution points uniformly
    fprintf('Uniform sampling %d on microlens plane.\n',...
        Num_sensorPixel*Num_sensorPixel);
    xinput_pinhole = [pinhole(:,1)-pinhole_R,pinhole(:,1)+pinhole_R];
    yinput_pinhole = [pinhole(:,2)+pinhole_R,pinhole(:,2)-pinhole_R];
    
    N_uniform_micro = Num_sensorPixel;
    N_angle = Num_sensorPixel;
    N_point = Num_sensorPixel;
    [uniformx_pinhole, uniformy_pinhole] = ...
        uniformCircleSample(xinput_pinhole,yinput_pinhole,N_angle,N_point);
    
    % random projection between sensor pixel and micro lens samples
    % (within each micro lens)
    const_vec = [0:N_uniform_micro*N_uniform_micro:N_uniform_micro*...
        N_uniform_micro*(pinhole_block-1)];
    const_mat = repmat(const_vec,N1*N2*N_uniform_micro*N_uniform_micro,1);
    ind_rand = randi(N_uniform_micro*N_uniform_micro,...
        N1*N2*sensorpixel_block,pinhole_block);
    ind_lcd_mat = ind_rand+const_mat;
    ind_lcd_micro = reshape(ind_lcd_mat,numel(ind_lcd_mat),1);
    
    inputx_pinhole = uniformx_pinhole(ind_lcd_micro);
    inputy_pinhole = uniformy_pinhole(ind_lcd_micro);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Project onto LCD plane %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    uc = transpose(reshape(transpose(repmat(pinhole(:,1),1,...
        N1*N2*sensorpixel_block)),1,N1*N2*sensorpixel_block*pinhole_block));
    vc = transpose(reshape(transpose(repmat(pinhole(:,2),1,...
        N1*N2*sensorpixel_block)),1,N1*N2*sensorpixel_block*pinhole_block));
    
    fprintf('Reverse ray tracing from the LCD plane.\n');
    % change to pinhole plane instead of the microlens plane
    [inttempx,inttempy] = intersection(jitterx,jittery,...
        inputx_pinhole,inputy_pinhole,...
        abs(LCD_z-pinhole_z),abs(LCD_z-obj_z));
    
    ind_in = 1:length(inttempx);
    ind_outx = find(inttempx>obj_bond(1,2) | inttempx<obj_bond(1,1));
    ind_outy = find(inttempy>obj_bond(2,2) | inttempy<obj_bond(2,1));
    ind_out = union(ind_outx,ind_outy);
    ind_in(ind_out) = [];
    
    % intersection point coordinates
    inxy = [inttempx(ind_in),inttempy(ind_in)];
    % find the k nearest neighbors in the projected image plane and assign
    % the average color to the sensor pixel
    
    nearK_img_ind_temp = ceil(inxy/obj_res);
    nearK_img_ind_temp(:,1) = nearK_img_ind_temp(:,1)+img_sizex/2;
    nearK_img_ind_temp(:,2) = nearK_img_ind_temp(:,2)-img_sizex/2-1;
    nearK_img_ind = (abs(nearK_img_ind_temp(:,2))-1)*img_sizex+nearK_img_ind_temp(:,1);
    
    
    for channel = 1:size(imgVecTotal,2)
        imgVec = imgVecTotal(:,channel);
        obj(:,1) = imgVec;
        imgProjectColor = obj(:,1);
        
        fprintf('Assign color to LCD plane.\n');
        colorAssign = zeros(size(inxy,1),1);
        for j = 1:K_img
            colorAssign=colorAssign+imgProjectColor(nearK_img_ind(:,j));
        end
        colorAssignT = zeros(size(inttempx,1),1);
        colorAssignT(ind_in) = colorAssign;
        
        blockcolor = transpose(sum(reshape(colorAssignT,N1*N2,numel(colorAssignT)/(N1*N2)),1));
        sensorColor(:,4)=blockcolor/(N1*N2);
        
        displaySensor(:,:,channel) = pixelvecTomat(sensorColor(:,4), Num_sensorPixel, Num_pinhole);
        
        fprintf('Finish LCD simulator channel %d.\n', channel);
        clear blockcolor
        clear colorAssign
        clear colorAssignT
    end
    
    clear const_vec
    clear const_mat
    clear ind_rand
    clear ind_lcd_mat
    clear ind_lcd_micro
    clear nearK_img_ind_temp
    clear imgProjectx
    clear imgProjecty
    clear nearK_img
    clear nearK_dis
    
    namefile = sprintf('lightfield/Objectdepth%03dmm.png',obj_z_vec(mm));
    imwrite(displaySensor,namefile,'png')
end
%
fprintf('Finish LCD simulator.\n');
figure()
imshow(displaySensor)
namefig = sprintf('Projected Image displayed on the LCD plane with object depth at %03d mm',obj_z_vec(mm));
title(namefig,'fontsize',15)

%% Retinal Image Rendering
eye_depth = 200;
img_focal_view_z_vec = -20; % eye focus plane
eye_view_z = eye_depth;
eye_view_center_vec = [-300,0;-50,0;-10,0;-3,0;0,0;3,0;10,0;50,0;300,0];
eye_view_center = [-200,0];
fprintf('Locating eye at depth %f mm, at (%f,%f).\n',...
    eye_view_z,eye_view_center(1),eye_view_center(2));
eye_view_res = 0.1;   % fixed, human eye parameter
eye_retina_dis = 20;  % fixed, human eye parameter
eye_FOV = pi/3;
eye_d = 24;

for nn = 1:length(img_focal_view_z_vec)
    view_z = img_focal_view_z_vec(nn);
    fprintf('Locating eye focal plane at depth %f mm.\n',view_z);
    
    [img_focal_pos(1),img_focal_pos(2)] =...
        intersection(eye_view_center(1),eye_view_center(2),...
        LCD_x,LCD_y,eye_view_z-pinhole_z,eye_view_z-view_z);
    
    % finite aperture size, eye bonding 4 corners
    eye_aperture_bond = [eye_view_center(1)-eye_d/2,eye_view_center(2)+eye_d/2;...
        eye_view_center(1)+eye_d/2,eye_view_center(2)+eye_d/2;...
        eye_view_center(1)-eye_d/2,eye_view_center(2)-eye_d/2;...
        eye_view_center(1)+eye_d/2,eye_view_center(2)-eye_d/2];
    
    view_sizex = Num_pinhole;
    view_sizey = Num_pinhole;
    view_block = view_sizex*view_sizey;
    
    view_phyx = physSizex_sensorPixel*(eye_depth-view_z)/(eye_depth-LCD_z);
    view_phyy = physSizey_sensorPixel*(eye_depth-view_z)/(eye_depth-LCD_z);
    
    view_res = view_phyx/view_sizex;
    
    finite_aperture_img = zeros(view_sizex,view_sizey,size(imgVecTotal,2));
    % set up eye image plane matrix and world coordinate
    imgDisplay = zeros(view_sizex*view_sizey,3);
    
    [X_eye,Y_eye] = meshgrid(1:view_sizex,1:view_sizey);
    x_eye_vec = (X_eye-1-view_sizex/2)*view_res+view_res/2+img_focal_pos(1);
    y_eye_vec = (-Y_eye+1+view_sizey/2)*view_res-view_res/2+img_focal_pos(2);
    imgDisplay(:,2:3) = [reshape(x_eye_vec',numel(x_eye_vec),1),...
        reshape(y_eye_vec',numel(y_eye_vec),1)];
    
    %%%%%%%%%%%%%%%%%%%%%% Interpolated image plane %%%%%%%%%%%%%%%%%%%%%%
    % cast multiple rays within each pixel on the image plane (focal plane)
    N_eye_focal = 16; % number of interpolate points--> trade-off with K_micro and K_LCD
    interpimg = zeros(size(imgDisplay,1)*N_eye_focal,3);
    interpimg(1:N_eye_focal:end,:) = imgDisplay;
    
    const_vecx_img = imgDisplay(:,2)-view_res/2;
    const_vecy_img = imgDisplay(:,3)-view_res/2;
    const_matx_img = reshape(repmat(const_vecx_img',N_eye_focal,1),...
        numel(const_vecx_img)*N_eye_focal,1);
    const_maty_img = reshape(repmat(const_vecy_img',N_eye_focal,1),...
        numel(const_vecx_img)*N_eye_focal,1);
    ind_rand_img = view_res.*rand(N_eye_focal*view_block,2);
    interp_mat_img = ind_rand_img+[const_matx_img,const_maty_img];
    interpimg(:,2:3) = interp_mat_img;
    
    % cast rays from the eye to the image plane then to the LCD plane
    % using quadrilinear sampling on microlens and LCD plane
    K_micro = 1;
    K_LCD = 1;
    
    for channel = 1:size(imgVecTotal,2)
        fprintf('Processing channel %d.\n',channel);
        
        sensorColorsep = pixelmatTovec(displaySensor(:,:,channel),Num_sensorPixel,Num_pinhole);
        
        %%%%%%%%%%%%%%%%%%%%%%%%% sample eye as finite aperture %%%%%%%%%%%%%%%%%
        % number of point samples on the finite eye aperture
        % uniformly divide the aperture into N_angle angles and within each angle,
        % randomly pick N_point points
        N_point = 5;
        N_angle = 5;
        fprintf('Sampling %d points on eye aperture.\n',N_point*N_angle)
        [uniformx_eye, uniformy_eye] = uniformCircleSample(eye_aperture_bond(1:2,1)',...
            eye_aperture_bond(2:3,2)',N_angle,N_point);
        display('Begin image rendering on the finite aperture')
        
        %%%%%%%%%%%% Rendering using sampled points on eye aperture %%%%%%%%%%%
        % iterate through all point samples on the aperture and then add up
        tic;
        for iter = 1:length(uniformx_eye)
            fprintf('Rendering in progress, currently processing the %dth point.\n',iter)
            % render image from the iter th sample on the eye aperture
            [inttempx,inttempy] = intersection(uniformx_eye(iter),uniformy_eye(iter),...
                interpimg(:,2),interpimg(:,3),...
                eye_view_z-view_z,eye_view_z-pinhole_z);
            
            ind_in = 1:length(inttempx);
            ind_outx = find(inttempx>bond_sensorPixel(1,2) | inttempx<bond_sensorPixel(1,1));
            ind_outy = find(inttempy>bond_sensorPixel(2,2) | inttempy<bond_sensorPixel(2,1));
            ind_out = union(ind_outx,ind_outy);
            ind_in(ind_out) = [];
            
            % intersection point coordinates on the micro lens plane
            inxy = [inttempx(ind_in),inttempy(ind_in)];
            inputx_pinhole = repmat(inttempx(ind_in),K_micro,1);
            inputy_pinhole = repmat(inttempy(ind_in),K_micro,1);
            
            %%%%%%%%%%%%%%%%%%%%%% LCD plane filtering %%%%%%%%%%%%%%%%%%%%%%
            % quadrilinear interpolation on micro-lens and LCD planes
            % 16 points (weighted according to distance) contribute to one rendered pixel
            
            % find the nearest K_micro micro lens on the micro lens plane
            % [nearK_micro_ind,nearK_micro_dis] = knnsearch(microLens,inxy,'k',K_micro);
            display('Looking for neareast microlens')
            nearK_micro_ind_temp = ceil(inxy/pinhole_pitch);
            nearK_micro_ind_temp(:,1) = nearK_micro_ind_temp(:,1)+Num_pinhole/2;
            nearK_micro_ind_temp(:,2) = nearK_micro_ind_temp(:,2)-Num_pinhole/2-1;
            nearK_micro_ind = (abs(nearK_micro_ind_temp(:,2))-1)*Num_pinhole+nearK_micro_ind_temp(:,1);
            microLensx = pinhole(:,1);
            microLensy = pinhole(:,2);
            nearK_micro = [microLensx(nearK_micro_ind),microLensy(nearK_micro_ind)];
            nearK_micro_dis = sqrt((inxy(:,1)-nearK_micro(:,1)).^2+(inxy(:,2)-nearK_micro(:,2)).^2);
            
            weight_micro_total = repmat(sum(nearK_micro_dis,2),1,K_micro);
            weight_nearK_micro = nearK_micro_dis./weight_micro_total;
            weight_nearK_micro_vec = reshape(weight_nearK_micro,numel(weight_nearK_micro),1);
            weight_nearK_micro_mat = repmat(weight_nearK_micro_vec,1,K_LCD);
            
            % input x,y locate on the chosen eye focal plane
            inputx_eye = repmat(interpimg(ind_in,2),K_micro,1);
            inputy_eye = repmat(interpimg(ind_in,3),K_micro,1);
            
            inputx_micro_center=[];
            inputy_micro_center=[];
            for i = 1:K_micro
                inputx_micro_center = [inputx_micro_center;pinhole(nearK_micro_ind(:,i),1)];
                inputy_micro_center = [inputy_micro_center;pinhole(nearK_micro_ind(:,i),2)];
            end
            
            % trace ray with the same incident angle but go through the center of
            % the nearest micro lenses
            inputx_eye_shift = inputx_eye+inputx_micro_center-inputx_pinhole;
            inputy_eye_shift = inputy_eye+inputy_micro_center-inputy_pinhole;
            % intersection at the LCD plane
            display('Tracing ray to the LCD plane')
            [intx_eye_LCD,inty_eye_LCD] = ...
                intersectwLens2(inputx_eye,inputy_eye,inputx_micro_center,...
                inputy_micro_center,inputx_micro_center,inputy_micro_center,...
                abs(view_z-pinhole_z),abs(view_z-LCD_z));
            
            indLCD_in = 1:length(intx_eye_LCD);
            indLCD_outx = find(intx_eye_LCD>bond_sensorPixel(1,2) | intx_eye_LCD<bond_sensorPixel(1,1));
            indLCD_outy = find(inty_eye_LCD>bond_sensorPixel(2,2) | inty_eye_LCD<bond_sensorPixel(2,1));
            indLCD_out = union(indLCD_outx,indLCD_outy);
            indLCD_in(indLCD_out) = [];
            
            % intersection point coordinates on the micro lens plane
            inx_eye_LCD = intx_eye_LCD(indLCD_in);
            iny_eye_LCD = inty_eye_LCD(indLCD_in);
            
            nearK_LCD_ind_temp = ceil([inx_eye_LCD,iny_eye_LCD]/sensorPixel_pitch);
            nearK_LCD_ind_temp(:,1) = nearK_LCD_ind_temp(:,1)+Num_pinhole/2*Num_sensorPixel;
            nearK_LCD_ind_temp(:,2) = nearK_LCD_ind_temp(:,2)-Num_pinhole/2*Num_sensorPixel-1;
            sensor_blockx = ceil(nearK_LCD_ind_temp(:,1)/Num_sensorPixel)-1;
            sensor_blocky = ceil(abs(nearK_LCD_ind_temp(:,2))/Num_sensorPixel)-1;
            modx = mod(nearK_LCD_ind_temp(:,1),Num_sensorPixel);
            modx(modx==0)=Num_sensorPixel;
            mody = mod(abs(nearK_LCD_ind_temp(:,2)),Num_sensorPixel);
            mody(mody==0)=Num_sensorPixel;
            nearK_LCD = sensor_blocky*sensorpixel_block*Num_pinhole+...
                sensor_blockx*sensorpixel_block+(mody-1)*Num_sensorPixel+modx;
            
            nearK_LCD_dis = sqrt((inx_eye_LCD-sensorPixel(nearK_LCD,2)).^2+...
                (iny_eye_LCD-sensorPixel(nearK_LCD,3)).^2);
            
            weight_LCD_total = repmat(sum(nearK_LCD_dis,2),1,K_LCD);
            weight_nearK_LCD = nearK_LCD_dis./weight_LCD_total;
            weight_LCD_micro = weight_nearK_micro_mat(indLCD_in).*weight_nearK_LCD;
            
            % weighted sum over all N_LCD sensor pixel intensities
            color_LCD_sum = zeros(size(nearK_LCD,1),1);
            for k = 1:K_LCD
                color_LCD_sum = color_LCD_sum + sensorColorsep(nearK_LCD(:,k)).*weight_LCD_micro(:,k);
            end
            
            % weighted sum over all N_micro micro lens intensities
            color_micro_sum = sum(reshape(color_LCD_sum,numel(color_LCD_sum)/K_micro,K_micro),2);
            interpimg(ind_in(indLCD_in),1) = color_micro_sum;
            
            % average over all sample points on the eye focal image plane
            for k = 1:size(imgDisplay,1)-1
                imgDisplay(k,1) = sum(interpimg((k-1)*N_eye_focal+1:k*N_eye_focal,1))/N_eye_focal;
            end
            imgDisplay(end,1) = interpimg(end,1);
            displayImg = transpose(reshape(imgDisplay(:,1), view_sizex, view_sizey));
            
            displayimagecell(:,:,iter) = displayImg;
        end
        toc
        
        % Superimpose all rendered image from sample points on the finite aperture
        finite_aperture_img(:,:,channel) = sum(displayimagecell,3)/length(uniformx_eye);
        
    end
    filename = [sprintf('focaldepth%03d',view_z) '.jpg'];
    imwrite(finite_aperture_img,filename)
    
    clear displayimagecell
end
%%%%%%%%%%%%%%%%%%%%% Visualization of rendered image %%%%%%%%%%%%%%%%%%%%%
figure()
imshow(finite_aperture_img)
title('Eye viewing of the display','fontsize',15)



