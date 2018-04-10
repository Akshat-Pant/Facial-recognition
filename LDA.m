%created functions for dimensionality reduction using LDA and classification using KNN
%accuracy metric also calculated
%the data is available in a 3d matrix as illum(1920, 21, 68).
%these are cropped images of 21 different subjects under 68 different illuminations.
%each image has already been converted to a vector.
%KNN is a non parametric technique.
%18 images of each person used for training. 3 images of each person used for testing.

clear;

load('/Users/akshatpant/Desktop/UMD/Sem 2/CMSC498M (ML)/Project1/Data/illumination.mat');
%create train data and test data
%train= illum(:, 1: 18, 1: 68);
%test= illum(:, 19: 21, 1: 68);

train= zeros(1920, 18, 68);
test= zeros(1920, 3, 68);
for j= 1: 68
    %randomize the order of illumination poses in each subject
    sub_matrix= illum(:, :, j);
    sub_matrix= sub_matrix(:, randperm(size(sub_matrix, 2)));
    for i= 1: 18
        train(:, i, j)= sub_matrix(:, i);
    end
    for i= 1: 3
        test(:, i, j)= sub_matrix(:, i+ 18);
    end
end
dim= 70;
[train_proj, test_proj]= LDA_imp(train, test, dim);

train_proj= reshape(train_proj, [dim, 18, 68]);
test_proj= reshape(test_proj, [dim, 3, 68]);

predictions= KNN(train_proj, test_proj, 3);

%calculate accuracy
count= 0;
for i= 1: 204
    if predictions(i)== ceil(i/3)
        count= count+ 1;
    end
end
accuracy= count/204;




function [proj_train, proj_test]= LDA_imp(train, test, n_dimensions)
    %datasets are 3d
    %outputs are 2d
    
    dim_train= size(train);
    dim_test= size(test);
    
    %reshape the dataset
    train= reshape(train, [dim_train(1), dim_train(2)*dim_train(3)]);
    test= reshape(test, [dim_test(1), dim_test(2)*dim_test(3)]);
    
    %calculate mean of all classes and all faces
    mean_face= mean(train, 2);
    class_means= zeros(dim_train(1), dim_train(3));
    for i= 1: dim_train(3)
        start= dim_train(2)*(i- 1)+ 1;
        stop= dim_train(2)*i;
        class_means(:, i)= mean(train(:, start: stop), 2);
    end
    
    %calculate SW within class scatter
    SW= zeros(dim_train(1), dim_train(1));
    for i= 1: dim_train(2)*dim_train(3)
        SW= SW+ (train(:, i)- class_means(:, ceil(i/dim_train(2))))*(train(:, i)- class_means(:, ceil(i/dim_train(2))))';
    end
    
    %calculate SB between class scatter
    SB= zeros(dim_train(1), dim_train(1));
    for i= 1: dim_train(3)
        SB= SB+ dim_train(2)*(class_means(i)- mean_face)*(class_means(i)- mean_face)';
    end
    
    %calculate the eigen faces
    [V, D]= eig(SB, SW);
    
    [D, I]= sort(diag(D), 'descend');      %sort the eigenvectors as per eigenvalues
    V= V(:, [I]);
    
    eigen_faces= V(:, 1: n_dimensions);     %select the lower dimensional space
    

    
    
    %project all training faces onto lower dimension
    proj_faces_train= [];
    for i= 1: dim_train(2)*dim_train(3)
        proj_faces_train= [proj_faces_train eigen_faces'* train(:, i)];
    end
    proj_train= proj_faces_train;
    
    %project all testing faces onto lower dimension
    proj_faces_test= [];
    for i= 1: dim_test(2)*dim_test(3)
        proj_faces_test= [proj_faces_test eigen_faces'*test(:, i)];
    end
    proj_test= proj_faces_test;
    
end





function class_predict= KNN(train, test, k)
    dim_train= size(train);
    dim_test= size(test);
    
    %reshape the data
    train= reshape(train, [dim_train(1), dim_train(2)*dim_train(3)]);
    test= reshape(test, [dim_test(1), dim_test(2)*dim_test(3)]);
    
    class_predict= [];
    for j= 1: dim_test(2)*dim_test(3)
        norms= [];
        for i= 1: dim_train(2)*dim_train(3)
            norms= [norms norm(test(:, j)- train(:, i))^2];
        end
        [norms, I]= sort(norms, 'ascend');
        knn= [];
        for i= 1: k
            knn= [knn ceil(I(i)/dim_train(2))];
        end
        class_predict= [class_predict mode(knn)];
    end

end
