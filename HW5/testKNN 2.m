%%%%%%
% This code is given and is not used in this assingment.
% It is, however, a useful matlab resource and I am therefore
% keeping it stored locally.
%
% Written By: Pengyu Hong
%%%%%%



oad KNN_Data;
plot( data1(:,1), data1(:,2), 'r.', data2(:,1), data2(:,2), 'b.', test(:,1), test(:,2), 'k^', 'MarkerSize', 10 );

%%
knnModel = ClassificationKNN.fit( [data1; data2], [ones(size(data1,1),1); 2*ones(size(data2,1),1)] );
y = predict(knnModel, test);
figure;
plot( data1(:,1), data1(:,2), 'r.', data2(:,1), data2(:,2), 'b.', test(:,1), test(:,2), 'k^', 'MarkerSize', 10 );
hold on;
if y == 1
    plot( test(:,1), test(:,2), 'r^' );
elseif y == 2
    plot( test(:,1), test(:,2), 'b^' );
end
title( ['K-NN: k = ', num2str(knnModel.NumNeighbors), ', Distance = ', knnModel.Distance] );
hold off;

%% Change the number of neighbors
knnModel.NumNeighbors = 3;
y = predict(knnModel, test);
figure;
plot( data1(:,1), data1(:,2), 'r.', data2(:,1), data2(:,2), 'b.', test(:,1), test(:,2), 'k^', 'MarkerSize', 10 );
hold on;
if y == 1
    plot( test(:,1), test(:,2), 'r^' );
elseif y == 2
    plot( test(:,1), test(:,2), 'b^' );
end
title( ['K-NN: k = ', num2str(knnModel.NumNeighbors), ', Distance = ', knnModel.Distance] );
hold off;

%% Use 'cosine' distance
knnModel = ClassificationKNN.fit( [data1; data2], [ones(size(data1,1),1); 2*ones(size(data2,1),1)], ...
                            'NSMethod','exhaustive', 'Distance','cosine');
knnModel.NumNeighbors = 1;
y = predict(knnModel, test);
figure;
plot( data1(:,1), data1(:,2), 'r.', data2(:,1), data2(:,2), 'b.', test(:,1), test(:,2), 'k^', 'MarkerSize', 10 );
hold on;
if y == 1
    plot( test(:,1), test(:,2), 'r^' );
elseif y == 2
    plot( test(:,1), test(:,2), 'b^' );
end
title( ['K-NN: k = ', num2str(knnModel.NumNeighbors), ', Distance = ', knnModel.Distance] );
hold off;

%% 
knnModel = ClassificationKNN.fit( [data1; data2], [ones(size(data1,1),1); 2*ones(size(data2,1),1)] );
knnModel.NumNeighbors = 10;  % 3 , 10
knnModel.DistanceWeight = 'inverse';
y = predict(knnModel, test);
figure;
plot( data1(:,1), data1(:,2), 'r.', data2(:,1), data2(:,2), 'b.', test(:,1), test(:,2), 'k^', 'MarkerSize', 10 );
hold on;
if y == 1
    plot( test(:,1), test(:,2), 'r^' );
elseif y == 2
    plot( test(:,1), test(:,2), 'b^' );
end
title( ['K-NN: k = ', num2str(knnModel.NumNeighbors), ', Distance = ', knnModel.Distance, ', DistanceWeight = ', knnModel.DistanceWeight] );
hold off;
