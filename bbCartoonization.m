function cartoonizedImage = bbCartoonization(humanImage)
    
    grayImage = rgb2gray(humanImage); % Converting image to grayscale

    % Applying bilateral filter for smoothing without blurring edges
    smoothedImage = imbilatfilt(grayImage, 'DegreeOfSmoothing', 5, 'SpatialSigma', 2);
    binaryImage = imbinarize(smoothedImage);  % Using adaptive thresholding to get binary image
    edgesImage = imcomplement(binaryImage); % Inverting the binary image to get edges

    % Combining the edges with the original image
    cartoonizedImage = humanImage;
    cartoonizedImage(repmat(edgesImage, [1, 1, 3])) = 0; % Seting edges to black
end
