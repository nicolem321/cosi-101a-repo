%% Load training data
fid = fopen( 'emnist-digits-train-images-idx3-ubyte', 'r' );
trainingImages = fread( fid, 'uint8' );
fclose( fid );
trainingImages = trainingImages(17 : end); % remove header (the first 16 bytes)
trainingImages = double(trainingImages) / 255;
trainingImages = reshape( trainingImages, 28, 28, 1, [] );

fid = fopen( 'emnist-digits-train-labels-idx1-ubyte', 'r' );
trainingLabels = fread( fid, 'uint8' );
fclose( fid );
trainingLabels = trainingLabels(9 : end); % remove header (the first 8 bytes)

%% Visualize training data
idx = randperm( 1000 ); 
idx = idx(1:20);
for k = 1 : 20
    subplot(4, 5, k);
    m = idx(k);
    s = m * 28 * 28 + 1;
    e = (m + 1) * 28 * 28;
    imshow( trainingImages( :,:,:,m) );
    title( num2str( trainingLabels(m) ) );
end

%% Load test data
fid = fopen( 'emnist-digits-test-images-idx3-ubyte', 'r' );
testImages = fread( fid, 'uint8' );
fclose( fid );
testImages = testImages(17 : end); % remove header (the first 16 bytes)
testImages = double(testImages) / 255;
testImages = reshape( testImages, 28, 28, 1, [] );

fid = fopen( 'emnist-digits-test-labels-idx1-ubyte', 'r' );
testLabels = fread( fid, 'uint8' );
fclose( fid );
testLabels = testLabels(9 : end); % remove header (the first 8 bytes)

%% Visualize test data
idx = randperm( 1000 ); 
idx = idx(1:20);
for k = 1 : 20
    subplot(4, 5, k);
    m = idx(k);
    s = m * 28 * 28 + 1;
    e = (m + 1) * 28 * 28;
    imshow( testImages(:, :, :, m) );
    title( num2str( testLabels(m) ) );
end

%%
aCNN = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 5, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer( [2, 2] )
    
    convolution2dLayer(3, 5)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer( [2, 2] )
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

%% Train Network
miniBatchSize  = 128;
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-3, ...
    'Plots','training-progress', ...
    'Verbose',false);
net = trainNetwork( trainingImages, categorical(trainingLabels), aCNN, options);

%% Apply the trained net to the test images
testPred = classify(net, testImages);
accuracy = sum(testPred == categorical(testLabels))/numel(testLabels);