% Loading input images
humanImage = imread("C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\Human Faces\1 (4).jpg");
cartoonRefImage = imread("C:\Users\shrav\OneDrive\Desktop\SIUE\Computer Vision\Cartoons\personai_icartoonface_rectest_0000184.jpg");

humanImage = imresize(humanImage, [224, 224]);  % adjusting size as needed
cartoonRefImage = imresize(cartoonRefImage, [224, 224]);
humanImage = double(humanImage) / 255.0;  % normalize to [0, 1]
cartoonRefImage = double(cartoonRefImage) / 255.0;

% Apply black-box cartoonization model
cartoonizedImage = blackBoxCartoonModel(humanImage, cartoonRefImage);

% Output images
figure;
subplot(1, 3, 1); 
imshow(humanImage); 
title('Human Face');
subplot(1, 3, 2); 
imshow(cartoonRefImage); 
title('Cartoon Reference');
subplot(1, 3, 3); 
imshow(cartoonizedImage); 
title('Cartoonized Image');
