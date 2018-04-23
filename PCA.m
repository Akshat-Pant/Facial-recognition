function [train_projected, test_projected]= PCA(train, test, n_dimensions)
    %datasets are 2-d
    %outputs are 2d
    
    dim_train= size(train);
    dim_test= size(test);
    
    %calculate mean of all faces
    mean_face= mean(train, 2);
    
    %subtract mean face from all faces
    norm_faces= [];
    for i= 1: dim_train(2)
        norm_faces= [norm_faces train(:, i)- mean_face];
    end
    
    %form a lower dimensional eigen face matrix
    C= norm_faces*norm_faces';
    [V, ~]= eigs(C, n_dimensions);   %extract m highest eigen faces
    
    eigen_faces= V;

    %project all training faces onto lower dimension
    proj_faces_train= [];
    for i= 1: dim_train(2)
        proj_faces_train= [proj_faces_train eigen_faces'* (train(:, i)- mean_face)];
    end
    train_projected= proj_faces_train;
    
    %project all testing faces onto lower dimension
    proj_faces_test= [];
    for i=1: dim_test(2)
        proj_faces_test= [proj_faces_test eigen_faces'*(test(:, i)- mean_face)];
    end
    test_projected= proj_faces_test;
    
end