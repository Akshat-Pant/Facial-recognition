load('/Users/akshatpant/Desktop/UMD/Sem 2/CMSC498M (ML)/Project1/Data/illumination.mat');
%create train data
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
        test(:, i, j)= sub_matrix(:, i+18);
    end
end
train= reshape(train, 1920, 18*68);
%test= reshape(test, 1920, 3*68);

%calculate mean of all faces
mean_face= mean(train, 2);

%subtract mean face from all faces
norm_faces= [];
for i= 1: 18*68
    norm_faces= [norm_faces train(:, i)- mean_face];
end

%form a lower dimensional eigen face matrix
C= norm_faces*norm_faces';
[V, D]= eigs(C, 700);   %select lower dimension space

eigen_faces= V;


%project all faces onto lower dimension
proj_faces= [];

for i= 1: 18*68
    proj_faces= [proj_faces eigen_faces'* train(:, i)];
end

test_face= test(:, 1, 50);

proj_test= eigen_faces'* test_face;

%compute k nearest neighbour algo
norms= [];
for i= 1: 18*68
    norms= [norms norm(proj_faces(:, i)- proj_test)^2];
end

min_norms= mink(norms, 3);  %select the value of k

KNN= [];
for i= 1: 18*68 
    if ismember(norms(i), min_norms)
        KNN= [KNN ceil(i/18)];
    end
end





