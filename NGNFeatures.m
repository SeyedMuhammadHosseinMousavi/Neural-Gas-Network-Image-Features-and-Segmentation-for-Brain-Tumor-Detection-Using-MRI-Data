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

% to double
for i = 1 : filesnumber(1,1)
X{i}=im2double(X{i});
disp(['double  :   ' num2str(i) ]);end;

% FA Contrast Adjustment
for i = 1 : filesnumber(1,1)
XX{i}=faenhance(X{i});
disp(['FA Contrast Adjust :   ' num2str(i) ]);end;

% gray
for i = 1 : filesnumber(1,1)
X{i}=rgb2gray(XX{i});
disp([' Gray :   ' num2str(i) ]);end;

% % Resize
for i = 1:filesnumber(1,1)   
X{i} = imresize(X{i},[64 64]);
disp(['Loading image No :   ' num2str(i) ]);
end;

% to vector
for i = 1 : filesnumber(1,1)
X{i} = X{i}(:)';;
disp(['Loading image No :   ' num2str(i) ]);
end;

%% Neural Gas Network (NGN) Parameters

ParVal.N = 2; % Number of Neurons 
ParVal.MaxIt = 5; % Number of runs

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


%%--------------------------------------------------
for i = 1 : filesnumber(1,1)
f{i} = sum(NGNnetwok{i}.w);
disp(['weight :   ' num2str(i) ]);
end;

clear d;
for i = 1 : filesnumber(1,1)
d(i,:)=f{i};
disp(['To Matrix :   ' num2str(i) ]);end;
clear f;
f=d; clear d;
temp=f;

hh=f(:,1:2:end-1);
% hh=hh(:,1:2:end-1);
% hh=hh(:,1:2:end-1);
% hh=hh(:,1:2:end-1);
% hh=hh(:,1:2:end-1);
% hh=hh(:,1:2:end-1);
% hh=hh(:,1:2:end-1);

NGNfeatures=hh;
sh=size(hh);sh=sh(1,2);
NGNfeatures(1:100,sh+1)=1;
NGNfeatures(101:200,sh+1)=2;

