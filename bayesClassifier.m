function predict= bayesClassifier(train, test)
    %train and test are 3d
    
    dim_train= size(train);
    dim_test= size(test);
    
    train= reshape(train, [dim_train(1), dim_train(2)*dim_train(3)]);
    test= reshape(test, [dim_test(1), dim_test(2)*dim_test(3)]);
    
    %calculate MLE of class means and class covariance
    class_means= [];
    class_cov= zeros(dim_train(1), dim_train(1), dim_train(3));
    for i= 1: dim_train(3)
        class_means= [class_means mean(train(:, dim_train(2)*(i-1)+ 1: dim_train(2)*i), 2)];
        classvar= train(:, dim_train(2)*(i-1)+ 1: dim_train(2)*i)- class_means(:, i);
        class_cov(:, :, i)= (1/dim_train(2))*(classvar*classvar'); 
    end
    
    %add small value to the diagonal to remove singularity
    for i= 1: dim_train(3)
        class_cov(:, :, i)= class_cov(:, :, i)+ eye(dim_train(1));
    end
    
    %calculate class probabilities
    predict= [];
    for i= 1: dim_test(2)*dim_test(3)
        class_prob= [];
        test_face= test(:, i);
        for j= 1: dim_train(3)
            scalar1= (2*pi)^(dim_train(1)/2);
            scalar2= det(class_cov(:, :, j))^0.5;
            scalar= 1/(scalar1*scalar2);
            exp1= test_face- class_means(:, j);
            exp2= pinv(class_cov(:, :, j));
            exponent= exp(-0.5* exp1'* exp2* exp1);
            class_prob= [class_prob scalar*exponent];
        end
        [~, I]= sort(class_prob, 'descend');
        predict= [predict I(1)];
    end
    
end