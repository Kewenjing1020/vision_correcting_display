% Ray Tracer of 2D image
%   with a specified focus depth, spatial and angular display resolution

% Created on Oct.5th
% by Xuaner Zhang
% Last revised: Nov.23rd

%% System setup
clc
close all
warning off
%%%%%%%%%%%%%%%%%%%%%% Image Specification %%%%%%%%%%%%%%%%%%%%%%
image = imread('test_image/0084.png');
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
Num_pinhole = 128; % spatical resolution
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
sensorPixel_R = sensorPixel_pitch/2;
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
sensorPixel_orig = sensorPixel;

%% Shear the light field to avoid cross-talk
% according to the depth of the eye
eye_depth = 200;
% find the shearing amount
[shearx,sheary] = intersection(0,0,pinhole(:,1),pinhole(:,2),...
    eye_depth-pinhole_z,eye_depth-LCD_z);
shearshiftx = shearx-sensorPixel((sensorpixel_block+1)/2:...
    sensorpixel_block:end,2);
shearshifty = sheary-sensorPixel((sensorpixel_block+1)/2:...
    sensorpixel_block:end,3);

% number of black pixels needed to be added
rel_num_blackx = sign(shearshiftx/sensorPixel_pitch).*...
    floor(abs(shearshiftx/sensorPixel_pitch));
rel_num_blacky = sign(shearshifty/sensorPixel_pitch).*...
    floor(abs(shearshifty/sensorPixel_pitch));
rel_matx = transpose(reshape(rel_num_blackx',Num_pinhole,Num_pinhole));
rel_maty = transpose(reshape(rel_num_blacky',Num_pinhole,Num_pinhole));

abs_matx = rel_matx;
abs_matx(:,1:Num_pinhole/2-1) = rel_matx(:,1:Num_pinhole/2-1)-...
    rel_matx(:,2:Num_pinhole/2);
abs_matx(:,Num_pinhole/2+1:end) = rel_matx(:,Num_pinhole/2+1:end)-...
    rel_matx(:,Num_pinhole/2:end-1);
insert_posx = find(abs_matx(1,:)>0);
insert_negx = sort(find(abs_matx(1,:)<0),'descend');

abs_maty = rel_maty;
abs_maty(1:Num_pinhole/2-1,:) = rel_maty(1:Num_pinhole/2-1,:)-...
    rel_maty(2:Num_pinhole/2,:);
abs_maty(Num_pinhole/2+1:end,:) = rel_maty(Num_pinhole/2+1:end,:)-...
    rel_maty(Num_pinhole/2:end-1,:);
insert_posy = sort(find(abs_maty(:,1)>0),'descend');
insert_negy = find(abs_maty(:,1)<0);

shearshiftx_mat = repmat(shearshiftx',sensorpixel_block,1);
shearshiftx_vec = reshape(shearshiftx_mat,numel(shearshiftx_mat),1);
shearshifty_mat = repmat(shearshifty',sensorpixel_block,1);
shearshifty_vec = reshape(shearshifty_mat,numel(shearshifty_mat),1);
shearshift = [shearshiftx_vec,shearshifty_vec];
sensorPixel = sensorPixel_orig;
sensorPixel(:,2:3) = sensorPixel(:,2:3)+shearshift;

%% LCD Simulator
% object depth
obj_z_vec = [20];

corner_LCD = sensorPixel(1,2);
corner_LCD2 = sensorPixel(Num_sensorPixel,2);
corner_pinhole = pinhole(1,1);

for mm = 1:length(obj_z_vec)
    obj_z = obj_z_vec(mm);
    
    [corner_objx,corner_objy] = intersection(0,0,pinhole(1,1)-pinhole_R,0,...
        eye_depth-pinhole_z,eye_depth-obj_z);
    
    physSizex_img = abs(corner_objx)*2;
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
    fprintf('Defining object resolution to be %f ppi.\n',25.4/display_img_res);
    
    % imgPlane
    %   1 col: intensity at the pixel
    %   2 col: x position in world coordinate
    %   3 col: y position in world coordinate
    
    displaySensor = zeros(Num_pinhole*Num_sensorPixel,...
        Num_pinhole*Num_sensorPixel,size(imgVecTotal,2));
    
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
        LCD_z-pinhole_z,LCD_z-obj_z);
    
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
    
    orig_displaySensor = displaySensor;
    black = zeros();
    for ninsert = 1:length(insert_posx)
        insert_sensor_posx = Num_sensorPixel*(insert_posx(ninsert)-1);
        insert_sensor_negx = Num_sensorPixel*(insert_negx(ninsert));
        insert_sensor_posy = Num_sensorPixel*(insert_posy(ninsert));
        insert_sensor_negy = Num_sensorPixel*(insert_negy(ninsert)-1);
        for ch = 1:3
            displaySensor_ch = displaySensor(:,:,ch);
            
            displaySensor_ch(:,insert_sensor_posx+1:Num_pinhole*Num_sensorPixel) =...
                displaySensor_ch(:,insert_sensor_posx:Num_pinhole*Num_sensorPixel-1);
            displaySensor_ch(:,insert_sensor_posx) = ...
                1/2*(displaySensor_ch(:,insert_sensor_posx-1)+displaySensor_ch(:,insert_sensor_posx+1));
            
            
            displaySensor_ch(:,1:insert_sensor_negx-1) =...
                displaySensor_ch(:,2:insert_sensor_negx);
            displaySensor_ch(:,insert_sensor_negx) =...
                1/2*(displaySensor_ch(:,insert_sensor_negx+1)+displaySensor_ch(:,insert_sensor_negx-1));
            
            displaySensor_ch(1:insert_sensor_posy-1,:) =...
                displaySensor_ch(2:insert_sensor_posy,:);
            displaySensor_ch(insert_sensor_posy,:) = 1/2*...
                (displaySensor_ch(insert_sensor_posy-1,:)+displaySensor_ch(insert_sensor_posy+1,:));
            
            displaySensor_ch(insert_sensor_negy+1:Num_pinhole*Num_sensorPixel,:) =...
                displaySensor_ch(insert_sensor_negy:Num_pinhole*Num_sensorPixel-1,:);
            displaySensor_ch(insert_sensor_negy,:) = 1/2*...
                (displaySensor_ch(insert_sensor_negy-1,:)+displaySensor_ch(insert_sensor_negy+1,:));
            
            displaySensor(:,:,ch) = displaySensor_ch;
        end
        insert_posx = insert_posx-1;
        insert_negx = insert_negx+1;
        insert_posy = insert_posy+1;
        insert_negy = insert_negy-1;
        
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
    
    %aber_power = diopterConvert([eye_depth/1000-obj_z_vec(mm)/1000,eye_depth/1000],0);
    namefile = sprintf('lightfield/BunnyLF.png');
    imwrite(displaySensor,namefile,'png')
end
%
fprintf('Finish LCD simulator.\n');
figure()
imshow(displaySensor)
namefig = sprintf('Light field.');
title(namefig,'fontsize',15)



