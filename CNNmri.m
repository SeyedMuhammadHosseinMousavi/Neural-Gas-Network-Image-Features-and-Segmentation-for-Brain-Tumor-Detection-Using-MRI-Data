
%% CNN 
deepDatasetPath = fullfile('surf');
imds = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 70;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([256 256 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',9, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation) *100;
disp(['CNN Recognition Accuracy Is =   ' num2str(accuracy) ]);
