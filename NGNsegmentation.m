% Image Segmentation and Quantization by Neural Gas Network (NGN)
% Define number of segments and iterations and get the output. 
% Org is image. You can use your image. 
% ParVal.N is Number of Segments
% ParVal.MaxIt is Number of runs
%----------------------------------------------------------------------
% clc;
% clear;
% close all;

%% Load Image

clc;
close all;
clear;
warning ('off');

%% Data Reading and Pre-Processing
path='mri3';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
X{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;
Org=X;

% to double
for i = 1 : filesnumber(1,1)
X{i}=im2double(X{i});
disp(['double  :   ' num2str(i) ]);end;

% FA Contrast Adjustment
for i = 1 : filesnumber(1,1)
XX{i}=faenhance(X{i});
disp(['FA Contrast Adjust :   ' num2str(i) ]);end;
FAenhanced=XX;

% gray
for i = 1 : filesnumber(1,1)
X{i}=rgb2gray(XX{i});
disp([' Gray :   ' num2str(i) ]);end;
tempx=X{1};
% % Resize
% for i = 1:filesnumber(1,1)   
% X{i} = imresize(X{i},[128 128]);
% disp(['Loading image No :   ' num2str(i) ]);
% end;

% to vector
for i = 1 : filesnumber(1,1)
X{i} = X{i}(:)';;
disp(['Loading image No :   ' num2str(i) ]);
end;

%% Neural Gas Network (NGN) Parameters

ParVal.N = 5; % Number of Segments
ParVal.MaxIt = 2; % Number of runs

ParVal.tmax = 10000;

ParVal.epsilon_initial = 0.3;
ParVal.epsilon_final = 0.02;
ParVal.lambda_initial = 2;
ParVal.lambda_final = 0.1;
ParVal.T_initial = 5;
ParVal.T_final = 10;

%% Training Neural Gas Network
for i = 1 : filesnumber(1,1)
NGNnetwok{i} = GasNN(X{i}, ParVal);
disp(['Loading image No :   ' num2str(i) ]);
end;

%% Vector to image and plot
for i = 1 : filesnumber(1,1)
Weight{i}=sum(round(rescale(NGNnetwok{i}.w,1,ParVal.N)));
disp(['weight :   ' num2str(i) ]);end;

for i = 1 : filesnumber(1,1)
Weight{i}=round(rescale(Weight{i},1,ParVal.N));
disp(['weight :   ' num2str(i) ]);end;

for i = 1 : filesnumber(1,1)
indexed{i}=reshape(Weight{i}(1,:),size(tempx));
disp(['indexed :   ' num2str(i) ]);end;

for i = 1 : filesnumber(1,1)
segmented{i} = label2rgb(indexed{i}); 
disp(['segment :   ' num2str(i) ]);end;


%% Plot Res
sample=1;

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
imshow(Org{sample},[]); title('Original');
subplot(2,2,2)
imshow(FAenhanced{sample},[]); title('FA');
subplot(2,2,3)
imshow(segmented{sample});
title(['Segmented in [' num2str(ParVal.N) '] Segments']);
subplot(2,2,4)
imshow(indexed{sample},[]);
title(['Quantized in [' num2str(ParVal.N) '] Thresholds']);

