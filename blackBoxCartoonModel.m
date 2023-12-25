function cartoonizedImage = blackBoxCartoonModel(humanImage, cartoonRefImage)
    grayHuman = rgb2gray(humanImage);  % Converting images to grayscale
    edgesHuman = edge(grayHuman, 'Canny'); % Edge detection using Canny
    cartoonQuantized = imquantize(cartoonRefImage, linspace(0, 1, 4));% Quantize colors in the cartoon reference image

    % Combining edges with the quantized cartoon reference image
    cartoonizedImage = humanImage;
    cartoonizedImage(repmat(edgesHuman, [1, 1, 3])) = cartoonQuantized(repmat(edgesHuman, [1, 1, 3]));
end
