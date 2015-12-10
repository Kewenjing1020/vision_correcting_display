% Ray Tracer of 2D image
% Created on Oct.5th
% by Xuaner Zhang
% Last revised: Dec.9th

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
micro_x = 0;
micro_y = 0;
FOV_micro = 5.51;
micro_z = FOV_micro;
display_res = 0.078; % resolution of the LCD plane
Num_micro = 128;
Num_LCD = 5; % directional resolution

micro_block = Num_micro*Num_micro;
micro_phy_size = Num_LCD*Num_micro*display_res;
physSizex_micro = micro_phy_size;
physSizey_micro = micro_phy_size;
micro_pitch = physSizex_micro/Num_micro;
micro_R = micro_pitch/2;
microLens = zeros(micro_block,2);

[micro_X,micro_Y] = meshgrid(1:Num_micro,1:Num_micro);
microx_vec = (micro_X-1-Num_micro/2)*micro_pitch+micro_pitch/2+micro_x;
microy_vec = (-micro_Y+1+Num_micro/2)*micro_pitch-micro_pitch/2+micro_y;
microLens(:,1:2) = [reshape(microx_vec',numel(microx_vec),1),...
    reshape(microy_vec',numel(microy_vec),1)];
%%%%%%%%%%%%%%%%%%%%%% LCD Plane %%%%%%%%%%%%%%%%%%%%%%
LCD_physSizex = physSizex_micro;
LCD_physSizey = physSizey_micro;
LCD_pitch = micro_pitch/Num_LCD;
LCD_R = LCD_pitch/2;
LCD_bond = [min(microLens(:,1)),max(microLens(:,1));...
    min(microLens(:,2)),max(microLens(:,2))];

LCD_x = 0;
LCD_y = 0;
LCD_z = 0;

LCD_block = Num_LCD*Num_LCD;
LCDplane = zeros(LCD_block,3);

[LCD_X,LCD_Y] = meshgrid(1:Num_micro*Num_LCD,1:Num_micro*Num_LCD);
LCDx_vec = (LCD_X-1-Num_micro*Num_LCD/2)*LCD_pitch+LCD_pitch/2+LCD_x;
LCDy_vec = (-LCD_Y+1+Num_micro*Num_LCD/2)*LCD_pitch-LCD_pitch/2+LCD_y;
LCD_xy_temp = [reshape(LCDx_vec',numel(LCDx_vec),1),...
    reshape(LCDy_vec',numel(LCDy_vec),1)];
LCDX_vec = reshape(LCD_X,numel(LCD_X),1);
LCDY_vec = reshape(LCD_Y,numel(LCD_Y),1);
LCD_blockx = ceil(LCDX_vec/Num_LCD)-1;
LCD_blocky = ceil(LCDY_vec/Num_LCD)-1;
LCDx_mod = mod(LCDX_vec,Num_LCD);
LCDx_mod(LCDx_mod==0)=Num_LCD;
LCDy_mod = mod(LCDY_vec,Num_LCD);
LCDy_mod(LCDy_mod==0)=Num_LCD;
LCD_index = [LCD_blockx*LCD_block*Num_micro+...
    LCD_blocky*LCD_block+(LCDx_mod-1)*Num_LCD+LCDy_mod]';
LCDplane(LCD_index,2:3) = LCD_xy_temp;
LCDplane(:,1) = reshape(repmat([1:micro_block],LCD_block,1),...
    LCD_block*micro_block,1);


%% LCD Simulator
% object depth
obj_z_vec = [20];

LCD_corner = LCDplane(1,2);
micro_corner = microLens(1,1)+micro_R;

for mm = 1:length(obj_z_vec)
    obj_z = obj_z_vec(mm);
    
    % set the object size
    corner_obj = LCD_corner+abs(micro_corner-LCD_corner)*...
        (micro_z-LCD_z)/(obj_z-micro_z);
    
    % set the object to be the size that all its values could be reached
    physSizex_img = 2*abs(corner_obj+micro_R);
    physSizey_img = physSizex_img;
    
    obj_res = physSizex_img/img_sizex;
    obj = zeros(img_sizex*img_sizey,3);
    [obj_X,obj_Y] = meshgrid(1:img_sizex,1:img_sizey);
    objx_vec = (obj_X-1-img_sizex/2)*obj_res+obj_res/2+img_x;
    objy_vec = (-obj_Y+1+img_sizey/2)*obj_res-obj_res/2+img_y;
    obj(:,2:3) = [reshape(objx_vec',numel(objx_vec),1),...
        reshape(objy_vec',numel(objx_vec),1)];
    
    obj_bond = [min(obj(:,2)),max(obj(:,2));...
        min(obj(:,3)),max(obj(:,3))];
    fprintf('Defining object resolution to be %f ppi.\n',25.4/display_res);
    
    % imgPlane
    %   1 col: intensity at the pixel
    %   2 col: x position in world coordinate
    %   3 col: y position in world coordinate
    
    displaySensor = zeros(Num_micro*Num_LCD,...
        Num_micro*Num_LCD,size(imgVecTotal,2));
    %     for channel = 1:size(imgVecTotal,2)
    %         imgVec = imgVecTotal(:,channel);
    %         obj(:,1) = imgVec;
    
    % LCD Simulator
    fprintf('Begin LCD Simulator.\n');
    K_img = 1; % k nearest neighbors (filtering parameter)
    % assume gray scale image
    sensorColor = zeros(size(LCDplane,1),size(LCDplane,2)+1);
    sensorColor(:,1:size(LCDplane,2)) = LCDplane;
    
    %%%%%%%%%%%%%%%% stratified sampling at LCD plane %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % within each sensor pixel, sample 25 points randomly using stratified
    % jittering algorithm
    N1 = 5;
    N2 = 5;
    xinput_LCD = [LCDplane(:,2)-LCD_R,LCDplane(:,2)+LCD_R];
    yinput_LCD = [LCDplane(:,3)+LCD_R,LCDplane(:,3)-LCD_R];
    
    fprintf('Jittered sampling 25 points on LCD plane.\n');
    [jitterx, jittery] = stratifiedSample(xinput_LCD,yinput_LCD,N1,N2);
    
    %%%%%%%%%%%%%%%% uniform sampling at micro-lens plane %%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % within each micro lens, sample #directional resolution points uniformly
    fprintf('Uniform sampling %d on microlens plane.\n',...
        Num_LCD*Num_LCD);
    xinput_micro = [microLens(:,1)-micro_R,microLens(:,1)+micro_R];
    yinput_micro = [microLens(:,2)+micro_R,microLens(:,2)-micro_R];
    
    N_uniform_micro = Num_LCD;
    N_angle = Num_LCD;
    N_point = Num_LCD;
    [uniformx_micro, uniformy_micro] = ...
        uniformCircleSample(xinput_micro,yinput_micro,N_angle,N_point);
    
    % random projection between sensor pixel and micro lens samples
    % (within each micro lens)
    const_vec = [0:N_uniform_micro*N_uniform_micro:N_uniform_micro*...
        N_uniform_micro*(micro_block-1)];
    const_mat = repmat(const_vec,N1*N2*N_uniform_micro*N_uniform_micro,1);
    ind_rand = randi(N_uniform_micro*N_uniform_micro,...
        N1*N2*LCD_block,micro_block);
    ind_lcd_mat = ind_rand+const_mat;
    ind_lcd_micro = reshape(ind_lcd_mat,numel(ind_lcd_mat),1);
    
    inputx_pinhole = uniformx_micro(ind_lcd_micro);
    inputy_pinhole = uniformy_micro(ind_lcd_micro);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Project onto LCD plane %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cx = transpose(reshape(transpose(repmat(microLens(:,1),1,...
        N1*N2*LCD_block)),1,N1*N2*LCD_block*micro_block));
    cy = transpose(reshape(transpose(repmat(microLens(:,2),1,...
        N1*N2*LCD_block)),1,N1*N2*LCD_block*micro_block));
    
    fprintf('Reverse ray tracing from the LCD plane.\n');
    % change to pinhole plane instead of the microlens plane
    [inttempx,inttempy] = intersectwLens(jitterx,jittery,...
        inputx_pinhole,inputy_pinhole,cx,cy,...
        LCD_z-micro_z,LCD_z-obj_z);
    
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
        
        displaySensor(:,:,channel) = pixelvecTomat(sensorColor(:,4), Num_LCD, Num_micro);
        
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
    
    namefile = sprintf('lightfield_micro/Object%dmm.png',obj_z_vec(mm));
    imwrite(displaySensor,namefile,'png')
end
%
fprintf('Finish LCD simulator.\n');
figure()
imshow(displaySensor)
namefig = sprintf('Light field of the object placed at %03d mm',obj_z_vec(mm));
title(namefig,'fontsize',15)


