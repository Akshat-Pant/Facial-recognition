clear;

addpath('/..../PCA.m');
addpath('/..../KNN.m');

load('/Users/akshatpant/Desktop/UMD/Sem 2/CMSC498M (ML)/Project1/Data/illumination.mat');

%create train data and test data
train= zeros(1920, 18, 68);
test= zeros(1920, 3, 68);
for j= 1: 68
    %randomize the order of illumination poses in each subject
    sub_matrix= illum(:, :, j);
    sub_matrix= sub_matrix(:, randperm(size(sub_matrix, 2)));
    for i= 1: 18
        train(:, i, j)= sub_matrix(:, i);   %select 18 images of each subject for training
    end
    for i= 1: 3
        test(:, i, j)= sub_matrix(:, i+18);     %select 3 images for testing
    end
end
train= reshape(train, 1920, 18*68);     %reshape data into a 2d matrix
test= reshape(test, [1920, 3*68]);

train= double(train)/255;
test= double(test)/255;


dim= 60;
[train_PCA, test_PCA]= PCA(train, test, dim);

train_PCA= reshape(train_PCA, [dim, 18, 68]);
test_PCA= reshape(test_PCA, [dim, 3, 68]);

predictions= KNN(train_PCA, test_PCA, 3);

accuracy= 0;
for j= 1: 68*3
    if predictions(j)== ceil(j/3)
        accuracy= accuracy+ 1;
    end
end

accuracy= accuracy/(68*3);
    
  



