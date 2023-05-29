%% Minimum Redundancy Maximum Relevance (MRMR) Algorithm.
%[idx,scores] = fsrmrmr(X,Y); % regression
[idx,scores] = fscmrmr(X,Y) % classification
%% Create a bar with the predictors importance
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score')
x=10;%1-max features
Filtfeat=idx(1:x);

%% Neighborhood Components analysis (NCA)
N=height (X);
%X_m=table2array(X);
% nca = fsrnca(X,Y,'FitMethod','exact','Solver','sgd','Standardize',true,...
%              'Lambda',0.0001,'Verbose',1,'NumPartitions',5); % regression
nca = fscnca(X,Y,'FitMethod','exact','Solver','sgd','Standardize',true,...
              'Lambda',0.0001,'Verbose',1,'NumPartitions',5);  % classification
%'LossFunction','mse',
figure()
plot(nca.FeatureWeights,'ro')
grid on
xlabel('Feature index')
ylabel('Feature weight')
sorted = sort(nca.FeatureWeights(:),'descend');
[~,index] = ismember(nca.FeatureWeights(:),sorted);
rankfeatures = reshape(index,size(nca.FeatureWeights));
XX=table(X_names',rankfeatures);
