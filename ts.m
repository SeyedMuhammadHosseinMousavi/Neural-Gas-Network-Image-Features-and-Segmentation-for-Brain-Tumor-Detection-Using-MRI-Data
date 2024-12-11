clear;
clc;
close all;
warning ('off');

%% Data Reading and Pre-Processing
path='y';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);
end;

% % Resize
for i = 1:filesnumber(1,1)   
sizee{i} = imresize(images{i},[256 256]);
disp(['Loading image No :   ' num2str(i) ]);
end;


%then saving new files
for i = 1 : filesnumber(1,1) 
imwrite(sizee{i},['p' num2str(i) '.jpg']);
    disp(['No of saved RGB image :   ' num2str(i) ]);
end;

