% Loading input images
imageDir = 'C:/Users/shrav/OneDrive/Desktop/SIUE/Computer Vision/Human Faces';
imds = imageDatastore(imageDir);

% Black Box Approach with Cartoon Renderer (using MATLAB's Cartoonizer)
outputDir = 'C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\MultiToonVision\cartoon_images';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%Applying Cartoonization to each image
for i = 1:length(imds.Files)
    img = imread(imds.Files{i});
    grayImg = rgb2gray(img); % Converting to grayscale  
    edgeImg = edge(grayImg, 'canny');  % Applying edge detection
    invertedEdgeImg = imcomplement(edgeImg); % Inverting the edges to get black lines on a white background
    
    % Combining original image and inverted edges to create cartoon effect
    cartoonizedImg = imfuse(img, invertedEdgeImg, 'blend', 'Scaling', 'joint');
    imwrite(cartoonizedImg, fullfile(outputDir, sprintf('cartoon_%04d.png', i)));
end

%White Box Approach (Edge Detection)
outputDir = 'C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\MultiToonVision\white_box_images';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%Applying edge detection to each image
for i = 1:length(imds.Files)
    img = imread(imds.Files{i});
    edgeImg = edge(rgb2gray(img), 'canny');
    imwrite(edgeImg, fullfile(outputDir, sprintf('white_box_%04d.png', i)));
end

% Neural Style Transfer
% Loading a pre-trained model for style transfer
net = vgg16;
styleLayer = 'relu1_1'; % Specifying the layer to extract features for style transfer
outputDir = 'C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\MultiToonVision\style_transfer_images';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for i = 1:length(imds.Files)
    img = imread(imds.Files{i});
    
    % Resizing the image to the input size of the pre-trained model
    imgResized = imresize(img, net.Layers(1).InputSize(1:2));
    contentActivation = activations(net, imgResized, styleLayer);
    styleActivation = activations(net, imgResized, styleLayer); % Extracting features from the style image
    stylizedImg = neuralStyleTransfer(imgResized, contentActivation, styleActivation);
    imwrite(stylizedImg, fullfile(outputDir, sprintf('style_transfer_%04d.png', i)));
end

function stylizedImg = neuralStyleTransfer(contentImg, contentFeatures, styleFeatures)
    transferNet = vgg16; % Creating a deep neural network for style transfer

    % Setting the layers for content and style representation
    contentLayer = 'relu4_2';
    styleLayers = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'};
    contentImg = resizeOrReshapeToMatch(contentImg, size(contentFeatures));
    contentImg = repmat(contentImg, [1, 1, 3]);
    contentActivation = activations(transferNet, contentImg, contentLayer);
    contentActivation = resizeOrReshapeToMatch(contentActivation, size(contentFeatures));

    contentLoss = sum((contentActivation - contentFeatures).^2);% Calculating the content loss
    styleLoss = 0; % Initializing the style loss
    for i = 1:length(styleLayers) % Calculating style loss for each style layer
        styleActivation = activations(transferNet, contentImg, styleLayers{i});

        % Resizing or reshaping styleActivation to match the size of styleFeatures
        styleActivation = resizeOrReshapeToMatch(styleActivation, size(styleFeatures(:, :, i)));

        gramStyle = gramMatrix(styleFeatures(:, :, i));
        gramContent = gramMatrix(styleActivation);
        layerStyleLoss = sum((gramContent - gramStyle).^2) / numel(gramStyle);
        styleLoss = styleLoss + layerStyleLoss;
    end

    % Combining content and style losses
    alpha = 1; % Adjusting the weight of content loss
    beta = 1e3; % Adjusting the weight of style loss
    totalLoss = alpha * contentLoss + beta * styleLoss;

    % Defining the optimization parameters
    options = optimset('fminunc');
    options.Display = 'off';

    % Optimizing the image to minimize the total loss
    stylizedImg = fminunc(@(x) computeLossAndGradient(x, transferNet, contentLayer, styleLayers, contentFeatures, styleFeatures, alpha, beta), double(contentImg(:)), options);

    % Reshaping the optimized image to its original size
    stylizedImg = reshape(stylizedImg, size(contentImg));
    
    % Rescaling the pixel values to the range [0, 255]
    stylizedImg = rescale(stylizedImg);
end

function [loss, gradient] = computeLossAndGradient(img, net, contentLayer, styleLayers, contentFeatures, styleFeatures, alpha, beta)

    contentActivation = activations(net, img, contentLayer);
    contentLoss = sum((contentActivation - contentFeatures).^2);
    styleLoss = 0;

    % Computing style loss for each style layer
    for i = 1:length(styleLayers)
        styleActivation = activations(net, img, styleLayers{i});
        gramStyle = gramMatrix(styleFeatures{i});
        gramContent = gramMatrix(styleActivation);

        layerStyleLoss = sum((gramContent - gramStyle).^2) / numel(gramStyle);
        styleLoss = styleLoss + layerStyleLoss;
    end

    % Combining content and style losses
    loss = alpha * contentLoss + beta * styleLoss;

    % Backward pass to compute the gradient
    gradient = gradientest(@(x) computeLossAndGradient(x, net, contentLayer, styleLayers, contentFeatures, styleFeatures, alpha, beta), img);
end

function gram = gramMatrix(features)
    % Computing the Gram matrix for a given set of features
    [h, w, c, n] = size(features);
    featuresReshaped = reshape(features, h * w, c, n);
    gram = zeros(c, c, n);
    for i = 1:n
        gram(:, :, i) = featuresReshaped(:, :, i)' * featuresReshaped(:, :, i) / (h * w);
    end
end

function resizedFeatures = resizeOrReshapeToMatch(features, targetSize)
    currentSize = size(features);
    
    % Checking if the size of features matches the target size
    if isequal(currentSize(1:2), targetSize(1:2))
        resizedFeatures = features;
    else
        resizedFeatures = imresize(features, targetSize(1:2));
    end
    
    % Checking if the number of channels matches
    if numel(currentSize) > 2 && numel(targetSize) > 2 && currentSize(3) ~= targetSize(3)
        % If not, replicating or reducing channels to match the target size
        if currentSize(3) < targetSize(3)
            resizedFeatures = repmat(resizedFeatures, [1, 1, targetSize(3)]); % Replicating channels
        else
            resizedFeatures = resizedFeatures(:, :, 1:targetSize(3)); % Reducing channels
        end
    end
end
