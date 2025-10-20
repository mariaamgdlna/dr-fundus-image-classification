clc;
clear;
close all;

datasetFolder = 'C:\Users\Maree\Downloads\Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02';

imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp(imds);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

disp(['Number of training images: ', num2str(numel(imdsTrain.Files))]);
disp(['Number of validation images: ', num2str(numel(imdsValidation.Files))]);

%% Function to extract GLCM features
function features = extractGLCMFeatures(img)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = imresize(img, [64 64]);
    
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(img, 'Offset', offsets);
    
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
end

%% Extract GLCM features from the training images
trainingFeatures = [];
trainingLabels = [];
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    features = extractGLCMFeatures(img);
    trainingFeatures = [trainingFeatures; features];
    trainingLabels = [trainingLabels; imdsTrain.Labels(i)];
end

%% Train the SVM classifier (using fitcecoc for multi-class)
SVMModel = fitcecoc(trainingFeatures, trainingLabels);

%% Extract GLCM features from the validation images
validationFeatures = [];
validationLabels = [];
for i = 1:numel(imdsValidation.Files)
    img = readimage(imdsValidation, i);
    features = extractGLCMFeatures(img);
    validationFeatures = [validationFeatures; features];
    validationLabels = [validationLabels; imdsValidation.Labels(i)];
end

%% Predict the labels of the validation images
predictedLabels = predict(SVMModel, validationFeatures);

%% Calculate the accuracy
accuracy = mean(predictedLabels == validationLabels);
disp(['Validation accuracy: ', num2str(accuracy * 100), '%']);

%% Display some sample predictions
idx = randperm(numel(imdsValidation.Files), 4);
figure;
for i = 1:4
    subplot(2, 2, i);
    I = readimage(imdsValidation, idx(i));
    imshow(I);
    label = predictedLabels(idx(i));
    title(string(label));
end

%% Evaluate the SVM classifier
accuracy = mean(predictedLabels == validationLabels);
disp(['Validation Accuracy: ', num2str(accuracy)]);

C = confusionmat(validationLabels, predictedLabels);

figure;
confusionchart(C, unique(validationLabels), 'Title', 'Confusion Matrix');

precision = diag(C) ./ (sum(C, 1) + (sum(C, 1) == 0)); % Add epsilon to denominator
recall = diag(C) ./ (sum(C, 2) + (sum(C, 2) == 0));
f1score = 2 * (precision .* recall) ./ (precision + recall + (precision + recall == 0));

fprintf('Precision: %.2f\n', mean(precision));
fprintf('Recall (Sensitivity): %.2f\n', mean(recall));
fprintf('F1 Score: %.2f\n', mean(f1score));

