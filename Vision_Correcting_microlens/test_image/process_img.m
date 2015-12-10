%% for gray scale images
filename = 'testimg_dog2';
fullfilename = strcat(filename,'.jpg');
newfilename = strcat(filename,'_square','.jpg');
startpixel = 450;

test = imread(fullfilename);
testgray = im2double(rgb2gray(test));
sizemin = min(size(testgray));
testgray_crop = imcrop(testgray,[startpixel,1,640-1,640-1]);

imwrite(testgray_crop,newfilename);

%% for color images
filename = 'testimg_rose';
fullfilename = strcat(filename,'.jpg');
newfilename = strcat(filename,'_square_color','.jpg');
startpixel1 = 1;
startpixel2 = 1;
test = imread(fullfilename);
testdouble = im2double(test);
sizemin = min(size(testdouble,1),size(testdouble,2));
testgray_crop = imcrop(testdouble,[startpixel1,startpixel2,640-1,640-1]);

img_size = 128;
testgray_resize = imresize(testgray_crop,[img_size,img_size]);
imwrite(testgray_resize,newfilename);

%%
filename = 'testimg_dog2';
