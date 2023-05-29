%% Predicting TCCs in the outlet using LSTM, ANN and RF
%% parameters required to be defined are as follows:
%% i) number of inputs ii)retention times iii) prediction horizon
%% iv)normalize data v)type of prediction vi) ML model  
%% vii) for classification define class limit 
clc
clear 
tic
%% Readtable with inputs and outputs and select parameters
WTW_data=readtable('Allhourlydata_new_maxmean.xlsx');
HRT=8; %select time between inlet and TW outlet 
SR_RT=11; %select SR retention time
PT=12; %select time horizon
WS=0; % size of the sliding window (1-5)/ 0 for no size window
mx=48; %day of maximum TCC prior to outlet in hours (24,48 etc.) 
parameters='All'; %All=all avaliable /outlet=only outlet TW=TW + outlet select=select parameters/ Nomax=all with no max-mean data % 'select'
par_selection=[3 2 6 16 4 1 23 14 18]; 
% 1=Inletflow 2=TurbInlet 3=pHInlet 4=ColourInlet 5=TWflow 6=TW_TCCs 7=TWPH 8=TWTurb 9=TWCl 
% 10=TWPhosphate 11=Outletflow 12=FinalPH 13=FinalTurb 14=FinalCl %
% 15=FinalAlum 16=FinalTCC 
prediction='regression'; %select prediction type (classification/regression)
cell_limit1=20000;cell_limit2=50000;cell_limit3=90000;    %set cell limit for class division 
Data='normalized'; % normalized/raw
ML_model='RF'; % ML method selection (LSTM/RF/ANN/BT/SVM/KNN)
partition='kfold'; %training/testing data separation (kfold/holdout/last10)
part_num=8; %which part is used for testing
output ='TCC'; %actual logarithmic TCCs (TCC/logTCC)

%% Inlet parameters
WTW_inlet_data=WTW_data(:,5:7); WTW_inlet_data=addvars(WTW_inlet_data,WTW_data.(1),WTW_data.(2),'Before','TurbInlet');
WTW_inlet_data.Properties.VariableNames{2} = 'Inletflow'; WTW_inlet_data=rmmissing(WTW_inlet_data);
%% TW parameters
WTW_TW_data=WTW_data(:,8:11); WTW_TW_data=addvars(WTW_TW_data,WTW_data.(1),WTW_data.(3),WTW_data.(16),'Before','TWPH');
WTW_TW_data.Properties.VariableNames{2} = 'TWflow'; WTW_TW_data.Properties.VariableNames{3} = 'TW_TCCs';
WTW_TW_data=WTW_TW_data(HRT+1:end,:); WTW_TW_data=rmmissing(WTW_TW_data,'DataVariables','TWflow');
%% Outlet parameters
WTW_outlet_data=WTW_data(:,[12:15,17]); WTW_outlet_data=addvars(WTW_outlet_data,WTW_data.(1),WTW_data.(4),'Before','FinalPH');
WTW_outlet_data.Properties.VariableNames{2} = 'Outletflow';
WTW_outlet_data=WTW_outlet_data(HRT+SR_RT+1:end,:);WTW_outlet_data=rmmissing(WTW_outlet_data,'DataVariables','Outletflow');
%% daily mean and max values 
WTW_data_max=WTW_data(HRT+SR_RT+PT+1:end,18:end); 
%% Outputs
TCCs=table(WTW_data.(1),WTW_data.(17));
TCCs=TCCs(HRT+SR_RT+PT+1:end,:);%TCCs=rmmissing(TCCs);

%% Sliding window daily max an min values and final input table
first_non_NAN = find(~isnan(TCCs.(2)),1); 
last_non_NAN = find(~isnan(TCCs.(2)),1,'last'); dist=height(TCCs)-last_non_NAN;
WTW_inlet_data_1=WTW_inlet_data(first_non_NAN:end-dist-HRT-SR_RT-PT,:);
WTW_TW_data_1=WTW_TW_data(first_non_NAN:end-dist-SR_RT-PT,:);
WTW_outlet_data_1=WTW_outlet_data(first_non_NAN:end-dist-PT,:);
WTW_data_max=WTW_data_max(first_non_NAN-mx:last_non_NAN-mx,:);
TCCs=TCCs(first_non_NAN:last_non_NAN,:);TCCs=TCCs(1:height(WTW_outlet_data_1),:);
if WS>0
    for i=1:WS
 
        WTW_inlet_data_SW{i}=WTW_inlet_data(first_non_NAN-i:end-dist-HRT-SR_RT-PT-i,:);
        for s=1:width(WTW_inlet_data_SW{i}); 
        names=append(WTW_inlet_data.Properties.VariableNames(s),"-",string(i),'hour');
        WTW_inlet_data_SW{i}=renamevars(WTW_inlet_data_SW{i},s,names);
        end

        WTW_TW_data_SW{i}=WTW_TW_data(first_non_NAN-i:end-dist-SR_RT-PT-i,:);
        for s=1:width(WTW_TW_data_SW{i}); 
        names=append(WTW_TW_data.Properties.VariableNames(s),"-",string(i),'hour');
        WTW_TW_data_SW{i}=renamevars(WTW_TW_data_SW{i},s,names);
        end

        WTW_outlet_data_SW{i}=WTW_outlet_data(first_non_NAN-i:end-dist-PT-i,:);
        for s=1:width(WTW_outlet_data_SW{i}); 
        names=append(WTW_outlet_data.Properties.VariableNames(s),"-",string(i),'hour');
        WTW_outlet_data_SW{i}=renamevars(WTW_outlet_data_SW{i},s,names);
        end
    end

    for i=1:WS
        WTW_inlet_data_1=[WTW_inlet_data_1 WTW_inlet_data_SW{1,i}(:,2:end)];
        WTW_TW_data_1=[WTW_TW_data_1 WTW_TW_data_SW{1,i}(:,2:end)];
        WTW_outlet_data_1=[WTW_outlet_data_1 WTW_outlet_data_SW{1,i}(:,2:end)];
    end
end
WTW_inlet_data=WTW_inlet_data_1;
WTW_TW_data=WTW_TW_data_1;
WTW_outlet_data=WTW_outlet_data_1;

clear names WTW_outlet_data_SW WTW_inlet_data_SW WTW_TW_data_SW
clear s WTW_outlet_data_1 WTW_inlet_data_1 WTW_TW_data_1 dist
clear first_non_NAN last_non_NAN 

%% Select parameters
switch parameters
    case 'All'
        X=[WTW_inlet_data(:,2:end) WTW_TW_data(:,2:end) WTW_outlet_data(:,2:end) WTW_data_max];
    case 'Nomax'
        X=[WTW_inlet_data(:,2:end) WTW_TW_data(:,2:end) WTW_outlet_data(:,2:end)];
    case 'outlet'
        X=WTW_outlet_data(:,2:end);  
    case 'TW'
        X=[WTW_TW_data(:,2:end) WTW_outlet_data(:,2:end)];
    case 'select'
         check=[WTW_inlet_data(:,2:end) WTW_TW_data(:,2:end) WTW_outlet_data(:,2:end) WTW_data_max];
         for i=1:length(par_selection)
             X(:,i)=check(:,par_selection(i));
             X_names (i) = check.Properties.VariableNames (par_selection(i));
         end
         X.Properties.VariableNames=X_names;
end
X_names=X.Properties.VariableNames;

switch output
    case 'TCC'
      Y=TCCs.(2);  
    case 'logTCC'
      Y=log10(TCCs.(2));
end

clear check HRT i  SR_RT WTW_inlet_data WTW_data_max
clear WTW_outlet_data WTW_TW_data TCCs 

%% cross validation
switch partition
    case 'kfold'
hpartition = cvpartition(Y,'KFold',20);
XYtemp=training(hpartition,part_num);
XYtesttemp=test(hpartition,part_num);
Xtrain=X(XYtemp,:); Xtest=X(XYtesttemp,:); wi=width(Xtrain);
Ytrain=Y(XYtemp,:); Ytest=Y(XYtesttemp,:);
clear XYtemp XYtesttemp
    case 'holdout'
hpartition = cvpartition(Y,'Holdout',0.005);
XYtemp=training(hpartition);
XYtesttemp=test(hpartition);
Xtrain=X(XYtemp,:); Xtest=X(XYtesttemp,:); wi=width(Xtrain);
Ytrain=Y(XYtemp,:); Ytest=Y(XYtesttemp,:);
clear XYtemp XYtesttemp
    case 'last10'
Xtrain=X(1:end-10,:); Xtest=X(end-9:end,:);wi=width(Xtrain);
Ytrain=Y(1:end-10,:); Ytest=Y(end-9:end,:);        
end

clear partition

%% select prediction type
 switch prediction
     case 'regression'
         Ytrain=Ytrain;
         Ytest=Ytest;
     case 'classification'
        Ytrain(Ytrain(:,1)<cell_limit1)=0; Ytrain(Ytrain(:,1)>=cell_limit1 & Ytrain(:,1) <cell_limit2)=1; 
        Ytrain(Ytrain(:,1)>=cell_limit2 & Ytrain(:,1) <cell_limit3)=2; Ytrain(Ytrain>=cell_limit3)=3;
        
        Ytest(Ytest(:,1)<cell_limit1)=0; Ytest(Ytest(:,1)>=cell_limit1 & Ytest(:,1) <cell_limit2)=1; 
        Ytest(Ytest(:,1)>=cell_limit2 & Ytest(:,1) <cell_limit3)=2; Ytest(Ytest>=cell_limit3)=3;
        
        uY=unique(Ytrain); %%count classes
        for iii=1:length(uY)
        cYtrain(iii,1)=sum(Ytrain==uY(iii));
        cYtest(iii,1)=sum(Ytest==uY(iii));
        end
        Ytrain=categorical(Ytrain); Ytest=categorical(Ytest);
 end
clear cell_limit1 cell_limit2 cell_limit3 iii uY
 
%% Normalize data
[Xtrain, Ytrain, Xtest,Ytest,Ytestn,m,s,mY,sY]=norm(Xtrain,Xtest,Ytrain,wi,Data,prediction,Ytest);
    
%% Train the ML model and make predictions
[Ypred, options,layers]=mlselection(Xtrain,Ytrain,Xtest,ML_model,prediction,wi,X_names);
 
%% Unnormalize prediction
switch prediction
    case 'regression'
 switch Data
    case 'normalized'
      Ypredn=Ypred;
      Ypredp=sY*Ypred+mY;
    case 'raw'
      Ypredp=Ypred;
      Ypredn=0;
 end
    case 'classification'
      Ypredp=Ypred;
end
clear Data mY sY wi hpartition
%% Model evaluation for regression or classification
switch prediction
    case 'regression' 
MAE = mae(Ytest,Ypredp);
fprintf('MAE of the predicted event = %g\n\n',MAE);
for iii=1:length(Ypredp)
ss(iii)=abs((Ytest(iii)-Ypredp(iii))/Ytest(iii));
end
MAPE = sum(ss)/length(Ytest);
fprintf('MAPE of the predicted event = %g\n\n',MAPE);
MSEn = mse(Ypredn,Ytestn);
fprintf('MSE of the normalized event = %g\n\n',MSEn);
RMSEn = sqrt(MSEn);
fprintf('RMSE of the normalized event = %g\n\n',RMSEn);
MSEp = mse(Ypredp,Ytest);
fprintf('MSE of the predicted event = %g\n\n',MSEp);
RMSEp = sqrt(MSEp);
fprintf('RMSE of the predicted event = %g\n\n',RMSEp);
NMSE = MSEp/var(Ytest);
fprintf('NMSE of the predicted event = %g\n\n',NMSE);
R2=(sum((Ytest(:,1)-mean(Ytest(:,1))).*(Ypredp(:,1)-mean(Ypredp(:,1))))/....
(sqrt(sum((Ytest(:,1)-mean(Ytest(:,1))).^2) .* sum((Ypredp(:,1)-mean(Ypredp(:,1))).^2))))^2;
fprintf('R2 of the predicted event = %g\n\n',R2);
NSE=1-(sum(((Ypredp(:,1)-Ytest(:,1)).^2))/(sum((mean(Ytest(:,1))-Ytest(:,1)).^2)));
fprintf('NSE of the predicted event = %g\n\n',NSE);
  case 'classification'
Ytest=double(Ytest)-1;
switch ML_model
    case 'RF'
    Ypred=str2double(Ypred);
    case 'BT'
    Ypred=Ypredp;
    case 'LSTM'
    Ypred=double(Ypredp)-1;
    case 'KNN'
    Ypred =double(Ypredp)-1;
    case 'SVM'
    Ypred =double(Ypredp)-1;
    case 'ANN'
    Ypred=Ypre
end
acc = sum(Ypred == Ytest)./numel(Ytest);
fprintf('Accuracy of the predicted event = %g\n\n',acc);
figure (2)
cm = confusionchart(Ytest,Ypred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
recall0=cm.NormalizedValues(1,1)/sum(cm.NormalizedValues(1,:));
recall1=cm.NormalizedValues(2,2)/sum(cm.NormalizedValues(2,:));
recall2=cm.NormalizedValues(3,3)/sum(cm.NormalizedValues(3,:));
recall3=cm.NormalizedValues(4,4)/sum(cm.NormalizedValues(4,:));
MR=(recall0+recall1+recall2+recall3)/4;
fprintf('High risk recall of the predicted event = %g\n\n',recall3)
fprintf('Macro - recall of the predicted event = %g\n\n',MR)
prec0=cm.NormalizedValues(1,1)/sum(cm.NormalizedValues(:,1));
prec1=cm.NormalizedValues(2,2)/sum(cm.NormalizedValues(:,2));
prec2=cm.NormalizedValues(3,3)/sum(cm.NormalizedValues(:,3));
prec3=cm.NormalizedValues(4,4)/sum(cm.NormalizedValues(:,4));
MP=(prec0+prec1+prec2+prec3)/4;
fprintf('Macro - precision of the predicted event = %g\n\n',MP)
MF0=2*(prec0*recall0)/(prec0+recall0);MF1=2*(prec1*recall1)/(prec1+recall1);
MF2=2*(prec2*recall2)/(prec2+recall2);MF3=2*(prec3*recall3)/(prec0+recall3);
MF=(MF0+MF1+MF2+MF3)/4;
fprintf('Macro - F1 Score of the predicted event = %g\n\n',MF)
end
   
%% write outputs
switch parameters
    case 'All'
file=table(Ytest,Ypredp);
file.Properties.VariableNames {2}=[ML_model '_' parameters '_' num2str(PT)];
filename=[ML_model '_' parameters '_winslide' num2str(WS) '_' num2str(PT) '_' num2str(part_num) '._rev1.csv'];
writetable(file,filename);

case 'Nomax'
file=table(Ytest,Ypredp);
file.Properties.VariableNames {2}=[ML_model '_' parameters '_' num2str(PT)];
filename=[ML_model '_' parameters '_winslide' num2str(WS) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv'];
writetable(file,filename);

   case 'select'
file=table(Ytest,Ypredp);
file.Properties.VariableNames {2}=[ML_model '_' parameters '_'...
    num2str(numel(par_selection)) '_'  num2str(PT)];
filename=[ML_model '_' parameters '_' num2str(numel(par_selection)) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv'];
writetable(file,filename);
end

%% Output graphs
switch prediction
    case 'regression'
figure
subplot(2,1,1)
plot(Ytest)
hold on
plot(Ypredp,'.-')
hold off
legend(["Observed TCCs" "Forecast TCCs"])
ylabel("TCC per ml")
title("Forecast")
subplot(2,1,2)
stem(Ypred - Ytest)
xlabel("Hour")
ylabel("Error")
title("RMSE = " + RMSEp)
% figurename=[ML_model '_' parameters '_' num2str(numel(par_selection)) '_' num2str(PT) '.bmp'];
% saveas (figure, figurename)
    case 'classification'

end
toc