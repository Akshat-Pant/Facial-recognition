function [proj_train, proj_test]= LDA(train, test, n_dimensions)
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
    
    [~, I]= sort(diag(D), 'descend');      %sort the eigenvectors as per eigenvalues
    V= V(:, [I]);
    
    eigen_faces= V(:, 1: n_dimensions);     %select the lower dimensional space
    
    %project all training faces onto lower dimension
    proj_faces_train= [];
    for i= 1: dim_train(2)*dim_train(3)
        proj_faces_train= [proj_faces_train eigen_faces'* (train(:, i)- mean_face)];
    end
    proj_train= proj_faces_train;
    
    %project all testing faces onto lower dimension
    proj_faces_test= [];
    for i= 1: dim_test(2)*dim_test(3)
        proj_faces_test= [proj_faces_test eigen_faces'*(test(:, i)- mean_face)];
    end
    proj_test= proj_faces_test;
    
end