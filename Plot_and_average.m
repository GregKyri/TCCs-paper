clear;
clc;
tic
%% Setting parameters inputs and readtable
method='KNN';
parameters='select'; % All / Nomax /select
par_selection=9;
WS=0;
PT=12;
part_num=15;
number_ML=2; %number of models
% filename=[method '_' parameters '_winslide' num2str(WS) '_' num2str(PT) '_' num2str(part_num) '.csv']
% filename=[method '_' parameters '_' num2str(par_selection) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv']
% filename=[method '_' parameters '_winslide' num2str(WS) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv']
KNN_filename=['KNN' '_' parameters '_' num2str(par_selection) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv']
RF_filename=['RF' '_' parameters '_' num2str(par_selection) '_' num2str(PT) '_' num2str(part_num) '_rev1.csv']

% T=readmatrix(filename);
RF=readmatrix(RF_filename); KNN=readmatrix(KNN_filename); 
%% remove negative values
RF=abs(RF); KNN=abs(KNN);
% T=abs(T);

%% Averaging models' outputs
All_par=[];
for i=1:length(KNN)
All_par(i,1)=KNN(i,1);
All_par(i,2)=(RF(i,2)+KNN(i,2))/number_ML; %average model
All_par(i,3)=0.4*RF(i,2)+0.6*KNN(i,2); %weighted average model
end

%% Average Model evaluation for regression
MAEav = mae(All_par(:,1)-All_par(:,2));
fprintf('MAE of the predicted event = %g\n\n',MAEav);
MSEpav = mse(All_par(:,1),All_par(:,2));
fprintf('MSE of the predicted event = %g\n\n',MSEpav);
RMSEpav = sqrt(MSEpav);
fprintf('RMSE of the predicted event = %g\n\n',RMSEpav);
CMr2av=(sum((All_par(:,1)-mean(All_par(:,1))).*(All_par(:,2)-mean(All_par(:,2))))/....
(sqrt(sum((All_par(:,1)-mean(All_par(:,1))).^2) .* sum((All_par(:,2)-mean(All_par(:,2))).^2))))^2;
fprintf('R2 of the predicted event = %g\n\n',CMr2av);
M=mean(All_par(:,1));
for i=1:length(All_par);
Supa(i)=(All_par(i,2)-All_par(i,1))^2;
Sdown(i)=(M-All_par(i,1))^2;
Supw(i)=(All_par(i,3)-All_par(i,1))^2;
end
NSE=1-(sum(Supa)/sum(Sdown));
NSEw=1-(sum(Supw)/sum(Sdown));
fprintf('NSE of the predicted event = %g\n\n',NSE)
%% Weighted Average Model evaluation for regression
MAEwav = mae(All_par(:,1)-All_par(:,3));
fprintf('MAEw of the predicted event = %g\n\n',MAEwav);
MSEpwav = mse(All_par(:,1),All_par(:,3));
fprintf('MSEw of the predicted event = %g\n\n',MSEpwav);
RMSEpwav = sqrt(MSEpwav);
fprintf('RMSEw of the predicted event = %g\n\n',RMSEpwav);
CMr2wav=(sum((All_par(:,1)-mean(All_par(:,1))).*(All_par(:,3)-mean(All_par(:,3))))/....
(sqrt(sum((All_par(:,1)-mean(All_par(:,1))).^2) .* sum((All_par(:,3)-mean(All_par(:,3))).^2))))^2;
fprintf('R2w of the predicted event = %g\n\n',CMr2wav);
fprintf('NSEw of the predicted event = %g\n\n',NSEw)

%% Plot each one separately
% Model_legend=method; 
% figure
% plot(T(:,1),'markers',8)
% hold on
% plot(T(:,2),'.-','markers',6)
% hold off
% legend("Observed TCCs",Model_legend)
% ylabel("TCCs per ml")
% title ( method,'TCCs Forecast')
% xlabel("Hour")

%% Plotting all together
RF_legend= ['R22']; KNN_legend= ['R23'];  Ave_legend=['AM'];
WAve_legend=['WAM'];
figure
plot(RF(:,1),'markers',8)
hold on
plot(RF(:,2),'.-','markers',6)
plot(KNN(:,2),'.-','markers',6)
plot(All_par(:,2),'.-','markers',6)
plot(All_par(:,3),'.-','markers',6)
hold off
legend("Observed TCCs",RF_legend,KNN_legend,Ave_legend,WAve_legend)
ylabel("TCCs per ml")
title("Single and combined ML models TCCs Forecast")
xlabel("Hour")

%% save file
All_par=[All_par RF(:,2) KNN(:,2)];
filename=['Average' '_' num2str(PT) '_' num2str(part_num) '.csv'];
writematrix(All_par,filename);
toc