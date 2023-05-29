function [options,layers] = OptimizeLSTMclass(X1,Y1,wi,DropoutLayer,BiLSTMLayer,numClasses)

k=randperm(size(X1,2));
X1traincheck=X1(:,k(1:4000)); Y1traincheck=Y1(:,k(1:4000));
X1testcheck=X1(:,k(4001:4500)); Y1testcheck=Y1(:,k(4001:4500));

opt.optimVars = [
    optimizableVariable('NumOfLayer',[1 3],'Type','integer')
    optimizableVariable('NumOfUnits',[50 200],'Type','integer')
    optimizableVariable('isUseBiLSTMLayer',[1 3],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];


   ObjFunc  = makeObjFcn(X1traincheck,Y1traincheck,X1testcheck,Y1testcheck);
    BayesObject = bayesopt(ObjFunc,opt.optimVars, ...
        'MaxTime',14*60*60, ...
        'IsObjectiveDeterministic',false, ...
        'MaxObjectiveEvaluations',40,...
        'Verbose',1,...
        'UseParallel',false)



function ObjFcn = makeObjFcn(Xtraincheck,Ytraincheck,X1testcheck,Y1testcheck)
ObjFcn = @valErrorFun;
function [valError,cons,fileName] = valErrorFun(optVars)
numFeatures = wi;
outputMode   = 'sequence';
numResponses = 1;
dropoutVal   = 0.5;
% if optVars.isUseBiLSTMLayer == 2
%     optVars.isUseBiLSTMLayer = 0;
% end
if DropoutLayer % if dropout layer is true
    if optVars.NumOfLayer ==1
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer==2
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer ==3
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer==4
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    end
    elseif optVars.NumOfLayer==5
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)                
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end

    else % if dropout layer is false
    if optVars.NumOfLayer ==1
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer ==2
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer ==3
        if BiLSTMLayer
            layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    elseif optVars.NumOfLayer ==4
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
       elseif optVars.NumOfLayer ==5
        if BiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
    end
end

opt.options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',optVars.InitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',40, ...
    'LearnRateDropFactor',0.1, ...
    'L2Regularization',optVars.L2Regularization, ...
    'Verbose',1, ...
    'MiniBatchSize',20,...
    'ExecutionEnvironment','cpu',...
    'Plots','training-progress');
disp('LSTM architect successfully created.');

%% --------------- Train small Network

try
    MLtraincheck = trainNetwork(X1traincheck,Y1traincheck,opt.layers,opt.options);
    disp('LSTM Netwwork successfully trained.');
    NetTrainSuccess =true;
catch me
    disp('Error on Training LSTM Network');
    NetTrainSuccess = false;
    return;
end
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
Ypredcheck=classify(MLtraincheck,X1testcheck,'MiniBatchSize',1);
valError =1-(sum(Ypredcheck == Y1testcheck)./numel(Y1testcheck));
Net  = MLtraincheck;
options = opt.options;
layers=opt.layers;
fieldName = ['ValidationError' strrep(num2str(valError),'.','_')];
if ismember('OptimizedParams',evalin('base','who'))
%     OptimizedParams =  evalin('base', 'OptimizedParams');
%     OptimizedParams.(fieldName).Net  = Net;
%     OptimizedParams.(fieldName).Opts = Opts;
    assignin('base','OptimizedParams',OptimizedParams);
else
%     OptimizedParams.(fieldName).Net  = Net;
%     OptimizedParams.(fieldName).Opts = Opts;
%     assignin('base','OptimizedParams',OptimizedParams);
end
fileName = num2str(valError) + ".mat";
% if opt.isSaveOptimizedValue
%     save(fileName,'Net','valError','Opts')
% end
cons = [];
end
end
end