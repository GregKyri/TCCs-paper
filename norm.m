function [Xtrain, Ytrain, Xtest,Ytest,Ytestn,m,s,mY,sY]=norm(Xtrain,Xtest,Ytrain,wi,Data,prediction,Ytest)

Xtrain=table2array(Xtrain);Xtest=table2array(Xtest);
[rows, columns] = find(isnan(Xtrain));
Xtrain(rows,:)=[];Ytrain(rows,:)=[];
[rows1, columns1] = find(isnan(Xtest));
Xtest(rows1,:)=[];Ytest(rows1,:)=[];
m = mean(Xtrain); 
s = std(Xtrain); 
writematrix(Xtest,'Testing_inputs.xlsx')
switch Data
    case 'normalized'
        Xtrain=normalize(Xtrain,'zscore');
    for i=1:size(Xtest,1)
    for j=1:wi
         Xtest(i,j) = (Xtest(i,j) - m(j)) / s(j);
    end
    end
      
switch prediction
    case 'regression'
        mY = mean(Ytrain); sY = std(Ytrain);
        Ytrain=normalize(Ytrain,'zscore');
    for ii=1:size(Ytest,1)
        Ytestn(ii,1) = (Ytest(ii,1) - mY) / sY;
    end
    case 'classification'
        mY=0; sY=0;
        Ytrain=Ytrain;
        Ytestn=Ytest;
end

    case 'raw'
        Xtrain=Xtrain;
        Ytrain=Ytrain;
        Xtest=Xtest;
        Ytest=Ytest;
        Ytestn=Ytest;
        m=0; s=0;mY=0;sY=0;
end
end
         