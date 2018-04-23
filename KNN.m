function class_predict= KNN(train, test, k)
    %train and test are both 3d
    %k is the number of nearest neighbours to compare

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
        [~, I]= sort(norms, 'ascend');
        knn= [];
        for i= 1: k         %extract the k nearest neighbours
            knn= [knn ceil(I(i)/ dim_train(2))];
        end
        class_predict= [class_predict mode(knn)];
    end

end