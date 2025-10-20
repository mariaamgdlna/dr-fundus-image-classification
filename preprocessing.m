imageFolder = 'D:\Dataset Diabetic Retinophaty\tambahan sever npdr';

if ~isfolder(imageFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s', imageFolder);
    uiwait(warndlg(errorMessage));
    return;
end

filePattern = fullfile(imageFolder, '**', '*.jpg'); % Change to whatever pattern you need
imageFiles = dir(filePattern);

outputDir = fullfile(imageFolder, 'output'); % Change to the desired output folder path

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for k = 1:length(imageFiles)
    baseFileName = imageFiles(k).name;
    fullFileName = fullfile(imageFiles(k).folder, baseFileName);
    
    relativePath = strrep(fullFileName, imageFolder, '');
    [relativeFolder, ~, ~] = fileparts(relativePath);
    outputFolder = fullfile(outputDir, relativeFolder);
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    try
        imageArray = imread(fullFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);
        
        % Display the image
        imshow(imageArray);
        title(baseFileName, 'Interpreter', 'none');
        drawnow; % Force display to update immediately
        
        % Perform your image processing here
        % grayImage = rgb2gray(imageArray);
        Resized_Image = imresize(imageArray, [584 565]);
        Converted_Image = im2double(Resized_Image);
        Lab_Image = rgb2lab(Converted_Image);
        fill = cat(3, 1, 0, 0);
        Filled_Image = bsxfun(@times, fill, Lab_Image);
        Reshaped_Lab_Image = reshape(Filled_Image, [], 3);
        [C, S] = pca(Reshaped_Lab_Image);
        S = reshape(S, size(Lab_Image));
        S = S(:, :, 1);
        Gray_Image = (S - min(S(:))) ./ (max(S(:)) - min(S(:)));
        Enhanced_Image = adapthisteq(Gray_Image, 'numTiles', [8 8], 'nBins', 128);
        Avg_Filter = fspecial('average', [9 9]);
        Filtered_Image = imfilter(Enhanced_Image, Avg_Filter);
        figure, imshow(Filtered_Image)
        title('Filtered Image')
        Subtracted_Image = imsubtract(Filtered_Image, Enhanced_Image);
        
        % Use graythresh as a placeholder for the Threshold_Level function
        level = graythresh(Subtracted_Image);
        
        Binary_Image = im2bw(Subtracted_Image, level - 0.008);
        figure, imshow(Binary_Image)
        title('Binary Image')
        
        Clean_Image = bwareaopen(Binary_Image, 100);
        figure, imshow(Clean_Image)
        title('Clean Image')
        
        Complemented_Image = imcomplement(Clean_Image);
        figure, imshow(Complemented_Image)
        title('Complemented Image')
        
        Final_Result = Colorize_Image(Resized_Image, Complemented_Image, [0 0 0]);
        figure, imshow(Final_Result)
        title('Final Result')
        
        outputFileName = fullfile(outputFolder, baseFileName);
        imwrite(Final_Result, outputFileName);
        
        % Feature Extraction using GLCM 
        Final_Result = imread(outputFileName);
        
        Gray_Final_Result = rgb2gray(Final_Result);
        
        % Compute the GLCM
        offsets = [0 1; -1 1; -1 0; -1 -1];
        GLCM = graycomatrix(Gray_Final_Result, 'Offset', offsets);
        
        % Compute GLCM properties
        GLCM_props = graycoprops(GLCM, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        
        % Display the GLCM properties
        disp('GLCM Properties:');
        disp(struct2table(GLCM_props));
        
    catch ME
        % If there is an error reading the image, display a warning and continue
        warning('Skipping file %s. Reason: %s', fullFileName, ME.message);
        continue;
    end
end

% Placeholder for the Colorize_Image function
% Replace this with the actual implementation of the Colorize_Image function
function colorizedImage = Colorize_Image(originalImage, binaryMask, color)
    colorizedImage = originalImage;
    for i = 1:3
        channel = originalImage(:,:,i);
        channel(binaryMask) = color(i);
        colorizedImage(:,:,i) = channel;
    end
end
