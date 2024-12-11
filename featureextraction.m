
clear;
clc;
close all;
warning ('off');

%% Data Reading and Pre-Processing
path='mri3';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;

% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
    si=size(images{i});
si=size(si);
    if si(1,2)==3;
grey{i}=rgb2gray(images{i});
disp(['To Gray :   ' num2str(i) ]);
    else 
      grey{i}=images{i};
    end;
end;


% Contrast Adjustment
for i = 1 : filesnumber(1,1)
images{i}=imadjust(grey{i});
disp(['Contrast Adjust :   ' num2str(i) ]);end;

% % Resize
% for i = 1:filesnumber(1,1)   
% images{i} = imresize(images{i},[512 512]);
% disp(['Loading image No :   ' num2str(i) ]);
% end;

% Sharp polished
% for i = 1 : filesnumber(1,1)
% [images{i},pic{i}]=sharppolished(images{i});
% disp(['Sharp polished  :   ' num2str(i) ]);end;

%% Feature Extraction

% Extract HOG Features 
clear HOG;
for i = 1 : filesnumber(1,1)
% The less cell size the more accuracy 
hog{i} = extractHOGFeatures(images{i},'CellSize',[64 64]);
disp(['Extract HOG :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
HOG(i,:)=hog{i};
disp(['HOG To Matrix :   ' num2str(i) ]);end;

% LPQ Features
clear LPQ_tmp;clear LPQ_Features;
winsize=19;
for i = 1 : filesnumber(1,1)
LPQ_tmp{i}=lpq(images{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
LPQ_Features(i,:)=LPQ_tmp{i};end;


% Load image data
imset = imageSet('surf','recursive'); 
% Extracting SURF features
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',200,...
    'PointSelection','Detector');
% Encode the images as new features
SURF = encode(bag,imset);



% Combining Feature Matrixes
% FinalReady=[HOG LPQ_Features SURF];

% FinalReady=[LPQ_Features];
% 
% FinalReady=[HOG];

FinalReady=[SURF];

% Labeling for Supervised Learning
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
FinalReady(1:100,sizefinal+1)=1;
FinalReady(101:200,sizefinal+1)=2;

