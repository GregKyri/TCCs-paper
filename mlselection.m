function [Ypred, options,layers]=mlselection(Xtrain,Ytrain,Xtest,ML_model,prediction,wi,X_names);

switch ML_model
    
    case 'LSTM' %training LSTM network
       numFeatures = wi;
       numResponses = 1;
       numClasses = numel(unique(Ytrain));
       numHiddenUnits = 50;
       DropoutLayer = true;
       BiLSTMLayer = true;
    switch prediction
    case 'regression'
      X1=Xtrain';   Y1=Ytrain';
%     X1=num2cell(Xtrain',[1 2]); 
%     Y1=num2cell(Ytrain',[1 2]);
%       [options,layers] = OptimizeLSTM(X1,Y1,wi,DropoutLayer,BiLSTMLayer)
%     Y1=Ytrain';
      layers = [ ...
      sequenceInputLayer(numFeatures)
       bilstmLayer(numHiddenUnits,'OutputMode','sequence')
       dropoutLayer(0.1)
%      bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%      dropoutLayer(0.1)
%      bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%      dropoutLayer(0.5)
%     bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%      lstmLayer(numHiddenUnits,'OutputMode','sequence')
%      lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];
    options=trainingOptions('adam', ...
      'MaxEpochs',150, ...
      'MiniBatchSize', 20, ...
      'GradientThreshold',1, ...
      'InitialLearnRate',0.001, ...
      'LearnRateSchedule','piecewise', ...
      'LearnRateDropPeriod',40, ...
      'LearnRateDropFactor',0.1 , ...
      'L2Regularization',0.000010721,...
      'Verbose',0, ...
      'Plots','training-progress');
    MLtrain=trainNetwork(X1,Y1,layers,options); %training LSTM network
    
    case 'classification'
    X1=Xtrain'; Y1=Ytrain';
%   [options,layers] = OptimizeLSTMclass(X1,Y1,wi,DropoutLayer,BiLSTMLayer,numClasses)
     layers = [ ...
     sequenceInputLayer(wi)
     bilstmLayer(numHiddenUnits,'OutputMode','sequence')
     dropoutLayer(0.1)
%      bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%      dropoutLayer(0.5)
%      bilstmLayer(numHiddenUnits,'OutputMode','sequence')
%      dropoutLayer(0.5)
%     bilstmLayer(numHiddenUnits,'OutputMode','sequence')
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer];
     options = trainingOptions('adam', ...
     'MaxEpochs',150, ...
     'GradientThreshold',1, ...
     'InitialLearnRate',0.012047, ...
     'LearnRateSchedule','piecewise', ...
     'LearnRateDropPeriod',40, ...
     'LearnRateDropFactor',0.1, ...
     'L2Regularization',9.59e-07, ...
     'Verbose',1, ...
     'MiniBatchSize',20,...
     'ExecutionEnvironment','cpu',...
     'Plots','training-progress');
 disp('LSTM architect successfully created.');
     MLtrain=trainNetwork(X1,Y1,layers,options); %training LSTM network
   end
    
    case 'RF' %training RF network
      Num_trees=1000;
      rng('default') 
      tallrng('default')
      MLtrain = TreeBagger(Num_trees,Xtrain,Ytrain,'Method',prediction,'NumPredictorsToSample',ceil(sqrt(wi)) ...
   ,'Surrogate','on','NumPrint',200,'OOBPrediction','on','PredictorSelection','curvature','OOBPredictorImportance','on');
  options=0; layers=0;
  
      case 'BT' %training AdaBoost network
      Ytrain=double(Ytrain)-1;
      Num_trees=1000;
      rng('default') 
      tallrng('default')
      N = length(Ytrain);  % Number of observations in the training sample
      t = templateTree('MaxNumSplits',N);
      MLtrain = fitcensemble(Xtrain,Ytrain,'Method','RUSBoost','NumLearningCycles',Num_trees,'Learners',t,...
     'LearnRate',0.1,'nprint',200);
      options=0; layers=0;
  
    case 'ANN'
       Ytrain=double(Ytrain);
       hiddenLayer=10;
       trainFcn = 'trainbr';
       performFcn='crossentropy';
       options=0; layers=0;    
      switch prediction
          
      case 'regression' %training ANN feedforward network
          MLtrain=feedforwardnet(hiddenLayer,trainFcn);
          MLtrain=train(MLtrain,Xtrain',Ytrain');
      case 'classification'
          MLtrain=patternnet(hiddenLayer,trainFcn,performFcn);
          MLtrain=train(MLtrain,Xtrain',Ytrain');
      end
    
    case 'SVM'
       switch prediction
           case 'regression' %training SVM for regression
               MLtrain = fitrsvm(Xtrain,Ytrain,'KernelFunction','gaussian','KernelScale','auto','Verbose',2);
               options=0; layers=0;
           case 'classification' %training SVM for multi - classiffication
               t = templateSVM('KernelFunction','gaussian');
              MLtrain = fitcecoc(Xtrain,Ytrain);%,'Learners',t,'Verbose',2,'Coding','onevsall')
              options=0; layers=0;
       end

    case 'KNN'   
       switch prediction
           case 'regression' %training KNN for regression
               k = 2;
               metric = 'euclidean';
               weights = {'distance'};
               mdl = kNNeighborsRegressor(k,metric,weights);
               mdl = mdl.fit(Xtrain,Ytrain);
               options=0; layers=0;
           case 'classification' %training KNN for multi - classiffication
               MLtrain=fitcknn(Xtrain,Ytrain,'NumNeighbors',4,'Distance','euclidean','DistanceWeight','squaredinverse');
                 options=0; layers=0;
       end
end

%% Predictions
    switch prediction
    case 'regression' 
    switch ML_model
    case 'LSTM'
    Ypred = predict(MLtrain,Xtest','MiniBatchSize',1);
    Ypred = Ypred';
    case 'RF' 
    Ypred = predict(MLtrain,Xtest);
    case 'SVM'
    Ypred = predict(MLtrain,Xtest);
    case 'ANN'
    Ypred=MLtrain(Xtest');
    Ypred = Ypred';
    case 'KNN'
    Ypred = mdl.predict(Xtest);
    Ypred = Ypred';
    end
    case 'classification' 
    switch ML_model
    case 'LSTM'
    Ypred = classify(MLtrain,Xtest');
    Ypred = Ypred';
    case 'RF' 
    Ypred = predict(MLtrain,Xtest);
    case 'BT' 
    Ypred = predict(MLtrain,Xtest);
    case 'SVM'
    Ypred = predict(MLtrain,Xtest);
    case 'KNN'
    Ypred = predict(MLtrain,Xtest);
    case 'ANN'
        Ypred = MLtrain(Xtest');
        Ypred = round(Ypred);
        Ypred = Ypred';
        Ypred = double(Ypred);
    end
    end
 
%% Plot predictors in RF or BT
switch ML_model
       case 'RF'
XX=categorical(X_names);XX=reordercats(XX,X_names);
imp = MLtrain.OOBPermutedPredictorDeltaError;
figure (1);
bar(XX,imp);
title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel =XX;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';
       case 'BT'
XX=categorical(X_names);XX=reordercats(XX,X_names);
imp = predictorImportance(MLtrain);
figure (1);
bar(XX,imp);
title('Curvature Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel =X_names;
h.XTickLabelRotation = 90;
h.TickLabelInterpreter = 'none';
end
save imp
end