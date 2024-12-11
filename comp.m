clear;
I=imread('d.jpg');
img=I;
I=rgb2gray(I);
% Adjust
I=imadjust(I);
% histeq
J = histeq(rgb2gray(img));
% Edge 
BW2 = edge(I,'canny');

% Otsu
thresh = multithresh(I,3);
seg_I = imquantize(I,thresh);
OtsuS = label2rgb(seg_I); 	

% watereshed
bw=imbinarize(I);
D = bwdist(~bw);
D = -D;
L = watershed(D);
% L(~bw) = 0;
% Water = label2rgb(L);
% imshow (L,[]);

% K-means
[L,Centers] = imsegkmeans(I,3);
KM = labeloverlay(I,L);

% imshow(img);
figure;
subplot(2,3,1)
imshow(BW2);title('Canny Edges','FontSize', 13,'color','b');
subplot(2,3,2)
imshow(I);title('Adjust Intensity','FontSize', 13,'color','b');
subplot(2,3,3)
imhist(I,32);title('Histogram Equalization','FontSize', 13,'color','b');
subplot(2,3,4)
imshow(OtsuS);title('Otsu','FontSize', 13,'color','b');
subplot(2,3,5)
imshow(L,[]);title('Watershed','FontSize', 13,'color','b');
subplot(2,3,6)
imshow(KM,[]);title('K-Means','FontSize', 13,'color','b');


