
% Loading input image
humanImage = imread("C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\Human Faces\1 (22).jpg");

humanImage = imresize(humanImage, [224, 224]);  % adjusting size as needed
humanImage = double(humanImage) / 255.0;  % normalize to [0, 1]

% Applying basic cartoonization model
cartoonizedImage = bbCartoonization(humanImage);

% Display and output results 
subplot(1, 2, 1); 
imshow(humanImage); 
title('Human Face');
subplot(1, 2, 2); 
imshow(cartoonizedImage); 
title('Cartoonized Image');

% Saving cartoonized image to a folder
outputFolder = "C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\Cartoon";
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

outputImagePath = fullfile(outputFolder, 'cartoonized_output.jpg');
imwrite(cartoonizedImage, outputImagePath);

disp(['Cartoonized image saved at: ' outputImagePath]);
