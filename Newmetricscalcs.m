%% calculate SI and KGE in the outputs
Output=readtable("SVM_Nomax_winslide0_12_5.csv");
%% SI calculation
MSEp = mse(Output.(2),Output.(1));
fprintf('MSE of the predicted event = %g\n\n',MSEp);
RMSEp = sqrt(MSEp);
fprintf('RMSE of the predicted event = %g\n\n',RMSEp);
SI=RMSEp/mean(Output.(1));
fprintf('SI of the predicted event = %g\n\n',SI);

%% calculate KGE
CC=corrcoef(Output.(2),Output.(1)); 
rm=mean(Output.(1)); %% mean of the observed values
rd=std(Output.(1)); %% std of the observed values
cm=mean(Output.(2)); %% mean of the predicted values
cd=std(Output.(2)); %% std of the predicted values
KGE=1-sqrt((CC(2)-1)^2+((cd/rd)-1)^2+((cm/rm)-1)^2);
fprintf('KGE of the predicted event = %g\n\n',KGE);

%% calculate ADR
a=Output.(2)./Output.(1);
ADR=sum(a)/length(Output.(1));
fprintf('ADR of the predicted event = %g\n\n',ADR);

%% calculate IOA
difpo=(Output.(2)-Output.(1)).^2;
difave=(abs(Output.(2)-mean(Output.(1)))+abs(Output.(1)-mean(Output.(1)))).^2;
IOA=1-(sum(difpo)/sum(difave));
fprintf('IOA of the predicted event = %g\n\n',IOA);

%% Calculate AICC  
k=106; %% 16 / 26 / 106   7 / 4 / 9
n=length(Output.(2));
AICC=n*log(sum(difpo)/n)+2*k+((2*k*(k+1)/(n-k-1)))+n*log(2*pi)+n;
%AICC=1+log(sum(difpo)/n)+((2*(k+1))/(n-k-2));
fprintf('AICC of the predicted event = %g\n\n',AICC);

%% Calculate PI
%Input=readtable("Testing_inputs.xlsx");
TCCsin=Input.(16);
difo=(TCCsin-Output.(1)).^2;
PI=1-(sum(difpo)/sum(difo));
fprintf('PI of the predicted event = %g\n\n',PI);