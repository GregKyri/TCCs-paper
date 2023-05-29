% function [options,layers] = OptimizeLSTM(Xtrain,Ytrain,wi)
% 
% optimVars = [
%     optimizableVariable('NumOfLayer',[1 5],'Type','integer')
%     optimizableVariable('NumOfUnits',[50 200],'Type','integer')
%     optimizableVariable('isUseBiLSTMLayer',[1 3],'Type','integer')
%     optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
%     optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
% 
% 
%     ObjFcn  = ObjFcn(optimVars,Xtrain,Ytrain);
%     BayesObject = bayesopt(ObjFcn,optimVars, ...
%         'MaxTime',14*60*60, ...
%         'IsObjectiveDeterministic',false, ...
%         'MaxObjectiveEvaluations',60,...
%         'Verbose',1,...
%         'UseParallel',false);
% end