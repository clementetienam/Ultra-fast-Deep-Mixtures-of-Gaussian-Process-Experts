clc;
clear ;
close all;
disp('@Author: Dr Clement Etienam')
disp('@Supervisor: Professor Kody Law')
disp('@Collaborator External : Professor Sara Wade')
disp('Ultra fast Deep Mixture of Gaussian Experts')
disp('Ths code is flexible and constructs a supervised learning model')
disp(' Three broader schemes are presented')
disp('1: Standard approaches void of any ensemble computation')
disp('2: Ensemble approach using Sparse Gp as experts and DNN as gates')
disp('3: 2 iteration variants of CCR-MM and random-MM with Gp experts and DNN gates')
disp('The sub five methods for constructing the machine are:')
disp('Method 1: Mixture of Experts model -GP experts and DNN gate')
disp('Method 2: Mixture of Experts model-MLP experts and DNN gate')
disp('Method 3: ML model using MLP alone')
disp('Method 4: Mixture of Experts model-DNN experts and DNN gate')
disp('Method 5: ML model using DNN alone')
disp('')
disp('*******************************************************************')
%%
disp('1=Standard Approach for Machine')
disp('2=Ensemble approach for machine')
disp('3= 2 iterations approach for machine-MM2r')
bigclement=input('Enter the option as stated above: ');
disp('*******************************************************************')
%%
if bigclement==1
disp('1=Reproduce results in paper using Gp experts and DNN gate')
disp('2=Try other combinations of experts and Gates')
Ultimate_Kody=input('Enter the option as stated above: ');
if Ultimate_Kody > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%%
if Ultimate_Kody==1
    Ultimate_clement=1;
else

disp('2=Mixture of experts with MLP experts and DNN gate')
disp('3=Machine learning model using MLP alone')
disp('4=Mixture of experts with DNN experts and DNN gate')
disp('5=Machine learning model using DNN alone')
Ultimate_clement=input('Enter the combinations of experts and Gates desired: ');
if (Ultimate_clement > 5) || (Ultimate_clement < 2)
error('Wrong choice please select 2-5')
end
disp('*******************************************************************')
end
switch Ultimate_clement
    case 1
 %%      
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp(' The experts are Gp and the Gate is a DNN')
disp('SET UP GPML TOOLBOX')
disp ('executing gpml startup script...')
mydir = fileparts (mfilename ('fullpath'));                 
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
addpath([mydir,'/util/sparseinv'])
addpath('CKS');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
      case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
      case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);   
    otherwise
            
        error('Data not specified correctly please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');  

    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of data points of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of data points of outputs is : %d \n',c);
%% Options for Training
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
disp('*******************************************************************')
%% Option to select the inducing points
disp('SELECT OPTION FOR INITIALISING INDUCING POINTS')
disp('1:K-means') % This throws an error sometimes
disp('2:Random')

method2=input('Enter the options for initialising inducing points: ');
if method2 > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Rescale data and then Split dataset to Train and Test;
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts')
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
    if Datause==1
        Experts=6;
    elseif Datause ==2
        Experts=7;
    elseif Datause ==3
    Experts=3;
    elseif Datause ==4
    Experts=2; 
    elseif Datause ==5
    Experts=4;
     elseif Datause ==6
    Experts=3; 
    elseif Datause ==7
    Experts=3;  
    elseif Datause ==8
    Experts=13; 
    elseif Datause ==9
    Experts=6; 
    elseif Datause ==10
    Experts=8; 
    elseif Datause ==11
    Experts=9; 
    elseif Datause ==12
    Experts=4;
    elseif Datause ==13
    Experts=3;     
    else
    Experts=3;
    end
        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
Experts=input('Enter the maximum number of experts required: ');
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') % This throws an error sometimes
disp('2:User preferred options')

choicee=input('Enter the options for setting the NN classifier parameters: ');
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
disp('*******************************************************************')    
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Classifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);

%% Choices used in the paper

end
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simulations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);


idx = kmeans(Data,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels 


diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{i,1}=hyper_updated;
    Xtrains{i,1}=Xuse;
    ytrains{i,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_tola,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')

[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

folder = 'Results_CCR';
mkdir(folder);
[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');

Matrix=[hardanswer,softanswer,stdclem];
headers = {'Hard_pred','Soft_pred','Stndev'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains)
cd(oldfolder)

elseif method==2
disp('CCR-MM SCHEME')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(Data,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ij,1}=hyper_updated;
    Xtrains{ij,1}=Xuse;
    ytrains{ij,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels              
[Valuee1,std1,cost3]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ik,1}=hyper_updated;
    Xtrains{ik,1}=Xuse;
    ytrains{ik,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels        
 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;
if abs(R2-R2now) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
%  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_CCR_MM';
mkdir(folder);
tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform(hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
Matrix=[hardanswer,softanswer,stdclem];
headers = {'Hard_pred','Soft_pred','Stndev'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains)
cd(oldfolder)
else
disp('*******************************************************************')    
  disp('MM-MM SCHEME') 
%  parpool('cluster1',8) 
tic;
 R2now=0; 
 meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
 disp('Initialised randomly for the first time')
else
weights=weights_updated;
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
disp('initialised using MM scheme')
end
diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{il,1}=hyper_updated;
    Xtrains{il,1}=Xuse;
    ytrains{il,1}=yuse;
 end
 %fprintf('Finished Expert %d... .\n', il);     
end

if i==1
[Valueeini,~,costini]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
fprintf('R2 initial accuracy for random initialisation is %4.4f... .\n', costini.R2);   
end

if i==1
dd_updated=dd;
else
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
end

[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels 

 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
% [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions ); 
%%           
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_MM_MM';
mkdir(folder);
tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform(hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte);
fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
Matrix=[hardanswer,softanswer,stdclem];
headers = {'Hard_pred','Soft_pred','Stndev'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains)
cd(oldfolder) 
end
disp('*******************************************************************')
rmpath('CKS')
rmpath(mydir)
disp('PROGRAM EXECUTED SUCCESFULLY')

    case 2
 %%
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp(' The experts are MLP and the Gate is a DNN')
addpath('CKS_DNN');
addpath('netlab');
addpath('data');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
           case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
          case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);  
    otherwise
            
        error('Data not specified correctly, please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% Options for Training
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
    if Datause==1
        Experts=13;
    elseif Datause ==2
        Experts=7;
    elseif Datause ==3
    Experts=6;
    elseif Datause ==4
    Experts=4; 
        elseif Datause ==5
        Experts=7;
         elseif Datause ==6
        Experts=5; 
        elseif Datause ==7
        Experts=4;  
        elseif Datause ==8
        Experts=13; 
        elseif Datause ==9
        Experts=12; 
        elseif Datause ==10
        Experts=16; 
    	elseif Datause ==11
    	Experts=9; 
    	elseif Datause ==12
    	Experts=4;
    	elseif Datause ==13
    	else
        Experts=13;
    end
        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS-USER SPECIFIC')
Experts=input('Enter the maximum number of experts required: ');
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') 
disp('2:User prefered options')

choicee=input('Enter the options for setting the NN classifier parameters: ');
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 3000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
disp('*******************************************************************')    
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Claasifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
end
%% Choices for MLP experts
disp('*******************************************************************')
disp('Choices for NN experts')
disp('1: User prefered options (As with the Paper)') %
disp('2: Pre-set  options')

kodyy=input('Enter the options for setting the  NN regressors: ');
if kodyy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
nin = size(X_train,2);			% input Layer.
if kodyy==1
mlp_nhidden = input('Enter the number of nodes required at Hiden layer,)use 500-1000: ');
else
  mlp_nhidden = 100;
end
dim_target = 1;			% Dimension of target space
alpha = 100;			% Inverse variance for weight initialisation
				% Make variance small for good starting point
options = foptions;
options(1) = 1;			% This provides display of error values.
options(14) = 3000; 

 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
idx = kmeans(Data,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels 


diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn(X_train,y_train,Classe);
 weights_updated{i,1}=net;

 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn(weights_updated,...
    dd_tola,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn(weights_updated,...
    dd_tola,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

folder = 'Results_CCR';
mkdir(folder);
[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 

fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder)

elseif method==2
disp('*******************************************************************')    
disp('CCR-MM SCHEME')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(Data,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);


% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn(X_train,y_train,Classe);    
weights_updated{ij,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('*******************************************************************')
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels              
[Valuee1,cost3]=prediction_clement_dnn(weights_updated,dd,X_train,...
    y_train,Class_all,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement_dnn(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn(X_train,y_train,Classe); 
weights_updated{ik,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement_dnn(weights_updated,X_train,y_train,modelNN,...
    Class_all,Experts);
[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
     
 [Valuee,cost2]=prediction_clement_dnn(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;
if abs(R2-R2now) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
%  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,updated_classtheta,nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_CCR_MM';
mkdir(folder);
tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn(weights_updated,...
    dd_updated,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn(weights_updated,dd_tola,...
    X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder)
else
disp('*******************************************************************')    
  disp('MM-MM SCHEME') 
%  parpool('cluster1',8) 
tic;
 R2now=0; 

%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
else
weights=weights_updated;
dd = MM_clement_dnn(weights,X_train,y_train,modelNN,Class_all,Experts); 
end

    [modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

% diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn(X_train,y_train,Classe);   
weights_updated{il,1}=net;

 end
 fprintf('Finished Expert %d... .\n', il);     
end

dd_updated = MM_clement_dnn(weights_updated,X_train,y_train,modelNN,...
    Class_all,Experts);

[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels 
 
 [Valuee,cost2]=prediction_clement_dnn(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
% [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions ); 
%%           
%oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_MM_MM';
mkdir(folder);
tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,costhardt]=prediction_clement_dnn(weights_updated,...
    dd_updated,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,costhard]=prediction_clement_dnn(weights_updated,...
    dd_tola,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);
fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder) 
end
rmpath('CKS_DNN')
rmpath('netlab')
rmpath('data')
disp('*******************************************************************')
disp('PROGRAM EXECUTED SUCCESFULLY')       
        
    case 3
 disp('BROAD OPTION OF FITTING A MODEL USING MLP ALONE')       
 disp('*******************************************************************')
addpath('CKS_MLP');
addpath('netlab');
addpath('data');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
   case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
  case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);       
    otherwise
            
        error('Data not specified correctly, please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% 
disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end

%% Experts options
disp('*******************************************************************')
%% Choices for MLP experts
disp('Choices for NN experts')
disp('1: User prefered options') %
disp('2: Pre-set  options')

kodyy=input('Enter the options for setting the  NN regressors: ');
if kodyy > 2
error('Wrong choice please select 1-2')
end
nin = size(X_train,2);			% input Layer.
if kodyy==1
disp('*******************************************************************')    
mlp_nhidden = input('Enter the number of nodes required at Hiden layer,)use 500-2000: ');
else
  mlp_nhidden = 500;
end
dim_target = 1;			% Dimension of target space
alpha = 100;			% Inverse variance for weight initialisation
				% Make variance small for good starting point
options = foptions;
options(1) = 1;			% This provides display of error values.
options(14) = 3000; 

 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
disp('*******************************************************************')
disp('Optimise experts')

tic;
[net]=optimise_experts(X_train,y_train);
tt=toc;
%% Prediction on Training data Training accuracy);

disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_mlp(net,X_train,y_train);
R2hardt=costhardt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);

%% Prediction on Test data (Test accuracy)

disp('predict Hard Prediction on test data')

[Valueehard,costhard]=prediction_clement_mlp(net,X_test,y_test);
R2hard=costhard.R2;


disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);


folder = 'Results';
mkdir(folder);
clem1=Plot_perform_mlp(hardtr,hardts,yini,folder,Xini,ind_train,ind_test,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
copyfile('Data_Summary.out',folder)

cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
Matrix=[clem1];
headers = {'Hard_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(net)
cd(oldfolder)
rmpath('CKS_MLP')
rmpath('netlab')
rmpath('data')
    case 4
   %%
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp('*******************************************************************')
disp(' The experts are DNN and the Gate is a DNN')
addpath('CKS_DNN');
addpath('netlab');
addpath('data');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
           case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
          case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);  
    otherwise
            
        error('Data not specified correctly, Please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% Options for Training
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
    if Datause==1
        Experts=6;
    elseif Datause ==2
        Experts=7;
    elseif Datause ==3
    Experts=3;
    elseif Datause ==4
    Experts=2; 
    elseif Datause ==5
    Experts=4;
     elseif Datause ==6
    Experts=3; 
    elseif Datause ==7
    Experts=3;  
    elseif Datause ==8
    Experts=13; 
    elseif Datause ==9
    Experts=6; 
    elseif Datause ==10
    Experts=8; 
    elseif Datause ==11
    Experts=9; 
    elseif Datause ==12
    Experts=4;
    elseif Datause ==13
    Experts=3;     
else
    Experts=3;
end
        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
Experts=input('Enter the maximum number of experts required: ');
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') 
disp('2:User prefered options')

choicee=input('Enter the options for setting the NN classifier parameters: ');
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Claasifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
end
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
idx = kmeans(Data,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels 


diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn_2(X_train,y_train,Classe);
 weights_updated{i,1}=net;

 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn_2...
    (weights_updated,dd_tola,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn_2...
    (weights_updated,dd_tola,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

folder = 'Results_CCR';
mkdir(folder);
[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 

fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder)

elseif method==2
disp('-------------------------CCR-MM SCHEME----------------------------')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(Data,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);


% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn_2(X_train,y_train,Classe);    
weights_updated{ij,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('*******************************************************************')
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels              
[Valuee1,cost3]=prediction_clement_dnn_2(weights_updated,dd,X_train,...
    y_train,Class_all,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement_dnn_2(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe); 
weights_updated{ik,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement_dnn_2(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
     
 [Valuee,cost2]=prediction_clement_dnn_2(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;
if abs(R2-R2now) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
                hiddenLayers,layers,updated_classtheta,nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_CCR_MM';
mkdir(folder);
tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn_2(weights_updated,...
    dd_updated,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn_2(weights_updated,dd_tola,...
    X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder)
else
disp('*******************************************************************')    
disp('-----------------------------MM-MM SCHEME---------------------------') 
%  parpool('cluster1',8) 
tic;
 R2now=0; 

%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
else
weights=weights_updated;
dd = MM_clement_dnn_2(weights,X_train,y_train,modelNN,Class_all,Experts); 
end

    [modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

% diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe);   
weights_updated{il,1}=net;

 end
 fprintf('Finished Expert %d... .\n', il);     
end

dd_updated = MM_clement_dnn_2(weights_updated,X_train,y_train,modelNN,Class_all,Experts);

[modelNN,~] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels 
 
 [Valuee,cost2]=prediction_clement_dnn_2(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
 [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
                hiddenLayers,layers,randInitializeWeights(layers),nnOptions ); 
%%           
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_MM_MM';
mkdir(folder);
tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,costhardt]=prediction_clement_dnn_2(weights_updated,...
    dd_updated,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,costhard]=prediction_clement_dnn_2(weights_updated,...
    dd_tola,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause);
fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy)
cd(oldfolder) 
end      
disp('*******************************************************************')
rmpath('CKS_DNN')
rmpath('netlab')
rmpath('data')
disp('---------------------PROGRAM EXECUTED SUCCESFULLY-------------------')       

    case 5
  disp('-----------BROAD OPTION OF FITTING A MODEL USING DNN ALONE--------')       
 disp('*******************************************************************')
addpath('CKS_MLP');
addpath('netlab');
addpath('data');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
   case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
  case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);       
    otherwise
            
        error('Data not specified correctly, please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% 
disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end

%% Experts options
disp('*******************************************************************')
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
disp('*******************************************************************')
disp('Optimise experts')

tic;

[net]=optimise_dnn(X_train,y_train);
tt=toc;
%% Prediction on Training data Training accuracy);

disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_dnn(net,X_train,y_train);
R2hardt=costhardt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);

%% Prediction on Test data (Test accuracy)

disp('predict Hard Prediction on test data')

[Valueehard,costhard]=prediction_dnn(net,X_test,y_test);
R2hard=costhard.R2;


disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);


folder = 'Results_DNN';
mkdir(folder);
clem1=Plot_perform_mlp(hardtr,hardts,yini,folder,Xini,ind_train,ind_test...
    ,oldfolder,Datause);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
copyfile('Data_Summary.out',folder)

cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
Matrix=[clem1];
headers = {'Hard_pred'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
parsave(net)
cd(oldfolder)  
rmpath('CKS_MLP')
rmpath('netlab')
rmpath('data')
    otherwise
 disp('METHOD NOT SPECIFIED CORRECTLY')         
end
elseif bigclement==2
    
    % ENSEMBLE APPROACH
    disp('--------Ensemble Mixture of Experts model -GP experts and DNN gate----------')
disp('')
disp('*******************************************************************')

%%

 %%      
disp('*******************************************************************')
disp(' The experts are Gp and the Gate is a DNN')
disp('SET UP GPML TOOLBOX')
disp ('executing gpml startup script...')
mydir = fileparts (mfilename ('fullpath'));                 
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
addpath([mydir,'/util/sparseinv'])
addpath('CKS_Ensemble');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
      case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
      case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);   
    otherwise
            
        error('Data not specified correctly, please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');  

    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% Options for Training
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 2
error('Wrong choice please select 1-3')
end
disp('*******************************************************************')
linecolor1 = colordg(4);
iterra=input('Enter the number of realisations you want: ');
%% Option to select the inducing points
disp('SELECT OPTION FOR INITIALISING INDUCING POINTS')
disp('1:K-means') % This throws an error sometimes
disp('2:Random')

method2=input('Enter the options for initialising inducing points: ');
if method2 > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Rescale data and then Split dataset to Train and Test;
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
  if Datause==1
    Experts=6;
elseif Datause ==2
    Experts=7;
elseif Datause ==3
Experts=3;
elseif Datause ==4
Experts=2; 
    elseif Datause ==5
    Experts=4;
     elseif Datause ==6
    Experts=3; 
    elseif Datause ==7
    Experts=3;  
    elseif Datause ==8
    Experts=13; 
    elseif Datause ==9
    Experts=6; 
    elseif Datause ==10
    Experts=8; 
    elseif Datause ==11
    Experts=9; 
    elseif Datause ==12
    Experts=4;
    elseif Datause ==13
    Experts=3;     
else
    Experts=3;
  end  
        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
Experts=input('Enter the maximum number of experts required: ');
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') % This throws an error sometimes
disp('2:User prefered options')

choicee=input('Enter the options for setting the NN classifier parameters: ');
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
disp('*******************************************************************')    
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Claasifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);

%% Choices used in the paper

end
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory

if method==1
    parfor mum=1:iterra
disp('*******************************************************************')          
fprintf('Starting realisation %d... .\n', mum);           
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);


idx = kmeans(Data,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );

[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels 


diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp paramters for experts
meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
for i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{i,1}=hyper_updated;
    Xtrains{i,1}=Xuse;
    ytrains{i,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_tola,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')

[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);


[hardanswer,softanswer,ind_trainn,ind_testn,stdclem]=Plot_perform(hardtr,softtr,hardts,softts,yini,...
    method,Xini,ind_train,ind_test,Datause,stdtr,stdte);
Hardbig(:,mum)=hardanswer;
Softbig(:,mum)= softanswer;
R2hardtrainingtola(mum,:)=R2hardt;
R2hardtesttola(mum,:)=R2hard;
R2softtrainingtola(mum,:)=R2softt;
R2softtesttola(mum,:)=R2soft;
Bigwallclock(mum,:)=tt;
stdbig(:,mum)=stdclem;
disp('*******************************************************************')  
fprintf('Finished realisation %d... .\n', mum);   
disp('*******************************************************************')  
    end


fprintf('The best realisation for hard R2 training accuracy is number %d with value %4.4f \n',find(R2hardtrainingtola == min(R2hardtrainingtola)),min(R2hardtrainingtola));
fprintf('The best realisation for hard R2 testing accuracy is number %d with value %4.4f \n',find(R2hardtesttola == min(R2hardtesttola)),min(R2hardtesttola));
fprintf('The best realisation for soft R2 training accuracy is number %d with value %4.4f \n',find(R2softtrainingtola == min(R2softtrainingtola)),min(R2softtrainingtola));
fprintf('The best realisation for soft R2 testing accuracy is number %d with value %4.4f \n',find(R2softtesttola == min(R2softtesttola)),min(R2softtesttola));
   
    
folder = 'Results_CCR';
mkdir(folder);    

cd(folder)

save('predict_hard.out','Hardbig','-ascii')
save('predict_soft.out','Softbig','-ascii')
save('R2hardtraining.out','R2hardtrainingtola','-ascii')
save('R2hardtesting.out','R2hardtesttola','-ascii')
save('R2softtraining.out','R2softtrainingtola','-ascii')
save('R2softtesting.out','R2softtesttola','-ascii')
save('Bigwallclock.out','Bigwallclock','-ascii')
save('Bigstd.out','stdbig','-ascii')
cd(oldfolder)

xx=[1:iterra]';

figure()
subplot(2,2,1)
plot(xx,R2hardtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR','location','northeast');

subplot(2,2,2)
plot(xx,R2hardtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR','location','northeast');

subplot(2,2,3)
plot(xx,R2softtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations')
title('R2 soft training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR','location','northeast');

subplot(2,2,4)
plot(xx,R2softtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 soft testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)

[Hard_mean,Hard_std,Soft_mean,Soft_std]=  Plot_Ensemble(folder,...
    oldfolder,Hardbig,Softbig,iterra,yini);
mumdad=[Hard_mean Hard_std Soft_mean Soft_std];
cd(folder)
save('UQ.out','mumdad','-ascii')
cd(oldfolder)

elseif method==2
R2evolve=cell(iterra,1); 
L2evolve=cell(iterra,1); 
RMSEevolve=cell(iterra,1);     
    for mum=1:iterra
disp('*******************************************************************')          
fprintf('Finished realisation %d... .\n', mum);           
disp('---------------------------CCR-MM SCHEME---------------------------')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(Data,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ij,1}=hyper_updated;
    Xtrains{ij,1}=Xuse;
    ytrains{ij,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels              
[Valuee1,~,cost3]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('begin iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ik,1}=hyper_updated;
    Xtrains{ik,1}=Xuse;
    ytrains{ik,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels        
 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;
if abs(R2-R2now) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
%  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory

tt=toc;

cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,Xini,ind_train,ind_test,Datause,stdtr,stdte);
Hardbig(:,mum)=hardanswer;
Softbig(:,mum)= softanswer;
R2hardtrainingtola(mum,:)=R2hardt;
R2hardtesttola(mum,:)=R2hard;
R2softtrainingtola(mum,:)=R2softt;
R2softtesttola(mum,:)=R2soft;
Bigwallclock(mum,:)=tt; 
stdbig(:,mum)=stdclem;
R2evolve{mum,1}=R2_allmm;
L2evolve{mum,1}=L2_allmm;
RMSEevolve{mum,1}=RMSE_allmm;
disp('*******************************************************************')  
fprintf('Finished realisation %d... .\n', mum);   
    end

fprintf('The best realisation for hard R2 training accuracy is number %d with value %4.4f \n',find(R2hardtrainingtola == min(R2hardtrainingtola)),min(R2hardtrainingtola));
fprintf('The best realisation for hard R2 testing accuracy is number %d with value %4.4f \n',find(R2hardtesttola == min(R2hardtesttola)),min(R2hardtesttola));
fprintf('The best realisation for soft R2 training accuracy is number %d with value %4.4f \n',find(R2softtrainingtola == min(R2softtrainingtola)),min(R2softtrainingtola));
fprintf('The best realisation for soft R2 testing accuracy is number %d with value %4.4f \n',find(R2softtesttola == min(R2softtesttola)),min(R2softtesttola));

    
    
folder = 'Results_CCR_MM';
mkdir(folder);
cd(folder)
save('predict_hard.out','Hardbig','-ascii')
save('predict_soft.out','Softbig','-ascii')
save('stdd.out','stdbig','-ascii')
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2hardtraining.out','R2hardtrainingtola','-ascii')
save('R2hardtesting.out','R2hardtesttola','-ascii')
save('R2softtraining.out','R2softtrainingtola','-ascii')
save('R2softtesting.out','R2softtesttola','-ascii')
save('Bigwallclock.out','Bigwallclock','-ascii')
cd(oldfolder)

xx=[1:iterra]';

figure()
subplot(2,2,1)
plot(xx,R2hardtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,R2hardtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,R2softtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations')
title('R2 soft training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,4)
plot(xx,R2softtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 soft testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)


[Hard_mean,Hard_std,Soft_mean,Soft_std]=  Plot_Ensemble(folder,oldfolder,...
    Hardbig,Softbig,iterra,yini);
mumdad=[Hard_mean Hard_std Soft_mean Soft_std];
cd(folder)
save('UQ.out','mumdad','-ascii')
cd(oldfolder)

figure()
for i=1:iterra

 subplot(2,2,1)
 a=R2evolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('R2 hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('R2 evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on
 
 subplot(2,2,2)
 a=L2evolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('L2 hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('L2 evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on 
 
  subplot(2,2,3)
 a=RMSEevolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('RMSE hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('RMSE evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on
 
end
hold off
cd(folder)
saveas(gcf,'performance_b.fig')
cd(oldfolder) 

cd(folder)
save('R2evolve.mat', 'R2evolve')
save('L2evolve.mat', 'L2evolve')
save('RMSEevolve.mat', 'RMSEevolve')
cd(oldfolder)
else
disp('*******************************************************************')    
  disp('-----------------------MM-MM SCHEME-------------------------------') 
%  parpool('cluster1',8) 
R2evolve=cell(iterra,1); 
L2evolve=cell(iterra,1); 
RMSEevolve=cell(iterra,1); 
for mum=1:iterra
fprintf('Starting realisation %d... .\n', mum);      
tic;
 R2now=0; 
 meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
%% Starting MM loop
 for i=1:Inf
fprintf('begin iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
 disp('Initialised randomly for the first time')
else
weights=weights_updated;
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
disp('initialised using MM scheme')
end
diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{il,1}=hyper_updated;
    Xtrains{il,1}=Xuse;
    ytrains{il,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', il);     
end

if i==1
[Valueeini,~,costini]=prediction_clement(weights_updated,dd,X_train,...
    y_train,Xtrains,ytrains,Experts);
fprintf('R2 initial accuracy for random initialisation is %4.4f... .\n', costini.R2);   
end

if i==1
dd_updated=dd;
else
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
end

[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels 

 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==50) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    fprintf('Finished iteration %d... .\n', i);          
 end
       
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform(hardtr,softtr,hardts,softts,yini,...
    method,Xini,ind_train,ind_test,Datause,stdtr,stdte);
Hardbig(:,mum)=hardanswer;
Softbig(:,mum)= softanswer;
R2hardtrainingtola(mum,:)=R2hardt;
R2hardtesttola(mum,:)=R2hard;
R2softtrainingtola(mum,:)=R2softt;
R2softtesttola(mum,:)=R2soft;
Bigwallclock(mum,:)=tt; 
stdbig(:,mum)=stdclem;
R2evolve{mum,1}=R2_allmm;
L2evolve{mum,1}=L2_allmm;
RMSEevolve{mum,1}=RMSE_allmm;
disp('********************************************************************')
fprintf('Finished realisation %d... .\n', mum);   
end
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_MM_MM';
mkdir(folder);
tt=toc;
fprintf('The best realisation for hard R2 training accuracy is number %d with value %4.4f \n',find(R2hardtrainingtola == min(R2hardtrainingtola)),min(R2hardtrainingtola));
fprintf('The best realisation for hard R2 testing accuracy is number %d with value %4.4f \n',find(R2hardtesttola == min(R2hardtesttola)),min(R2hardtesttola));
fprintf('The best realisation for soft R2 training accuracy is number %d with value %4.4f \n',find(R2softtrainingtola == min(R2softtrainingtola)),min(R2softtrainingtola));
fprintf('The best realisation for soft R2 testing accuracy is number %d with value %4.4f \n',find(R2softtesttola == min(R2softtesttola)),min(R2softtesttola));

cd(folder)
save('predict_hard.out','Hardbig','-ascii')
save('predict_soft.out','Softbig','-ascii')
save('stdd.out','stdbig','-ascii')
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2hardtraining.out','R2hardtrainingtola','-ascii')
save('R2hardtesting.out','R2hardtesttola','-ascii')
save('R2softtraining.out','R2softtrainingtola','-ascii')
save('R2softtesting.out','R2softtesttola','-ascii')
save('Bigwallclock.out','Bigwallclock','-ascii')
cd(oldfolder)

xx=[1:iterra]';

figure()
subplot(2,2,1)
plot(xx,R2hardtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,2)
plot(xx,R2hardtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 hard testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,3)
plot(xx,R2softtrainingtola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations')
title('R2 soft training accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,4)
plot(xx,R2softtesttola,'r','LineWidth',1)
xlim([1 iterra])
ylabel('R2 in %') 
xlabel('Realisations') 
title('R2 soft testing accuracy in % ')
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)

[Hard_mean,Hard_std,Soft_mean,Soft_std]=  Plot_Ensemble(folder,...
    oldfolder,Hardbig,Softbig,iterra,yini);
mumdad=[Hard_mean Hard_std Soft_mean Soft_std];
cd(folder)
save('UQ.out','mumdad','-ascii')
cd(oldfolder)


figure()
for i=1:iterra

 subplot(2,2,1)
 a=R2evolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('R2 hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('R2 evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on
 
 subplot(2,2,2)
 a=L2evolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('L2 hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('L2 evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on 
 
  subplot(2,2,3)
 a=RMSEevolve{i,:};
b=1:size(a,1);
 plot(b,a,'Color',linecolor1,'LineWidth',2)
 xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('RMSE hard training accuracy in %','FontName','Helvetica', 'Fontsize', 13);
title('RMSE evolution','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [a];
legend(h,'Realisations','location','northeast');
 hold on
 
end
hold off
cd(folder)
saveas(gcf,'performance_b.fig')
cd(oldfolder)  

end
disp('*******************************************************************')
rmpath('CKS_Ensemble')
rmpath(mydir) 

else
 disp('---------------------2 iterations random-MM /CCR-MM-----------------')   

    Ultimate_clement=1;


disp('*******************************************************************')

 %%      
disp('*******************************************************************')
disp(' The experts are Gp and the Gate is a DNN')
disp('SET UP GPML TOOLBOX')
disp ('executing gpml startup script...')
mydir = fileparts (mfilename ('fullpath'));                 
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
addpath([mydir,'/util/sparseinv'])
addpath('CKS');
oldfolder=cd;
%% Select the Data to use
disp('*******************************************************************')
disp('CHOOSE THE DATASET')
disp('')
disp('1=NASA data')
disp('2=Motorcycle data')
disp('3=Hingdom data')
disp('4=FODS 1 data')
disp('5=FODS 2 data')
disp('6=FODS 3 data')
disp('7=FODS 4 data')
disp('8=FODS 5 data')
disp('9=FODS 6 data')
disp('10=FODS 7 data')
disp('11=Tauth data')
disp('12=Liu data-Large-scale Heteroscedastic Regression via Gaussian Process')
disp('13=Mixture model 1 data')
disp('14=Mixture model 2 data')
Datause=input('Enter the dataset from 1-14 you want to simulate: ');
switch Datause
    case 1
        cd('data/Nasa')
        load ('nasadata.txt');
        aa=nasadata;
        X=aa(:,1:3);
        y=aa(:,4);
        cd(oldfolder)
    case 2
        cd('data/Motorcycle')
        load motorcycle;
        X=x;
        cd(oldfolder)
    case 3
        X = unifrnd(0,20,1000,1);
        for i=1:size(X,1)
            x=X(i,:);
            if (x < 10)
            aa = sin(pi*x/5) + 0.2*cos(4*pi*x/5);
            else
            aa = x/10 - 1;
            end
            y(i,:) = aa;
        end
        r = normrnd(0,0.1^2,1000,1);
        y=y+r; %corrrupt with noise
        cd('data/Hingdom')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
    case 4
        X=linspace(0,2,1000);
        X=X';
        y=X.*(X>=1);
        cd('data/FODS_1')
        file3 = fopen('inputt.out','w+'); 
        for k=1:numel(X)                                                                       
        fprintf(file3,' %4.8f \n',X(k) );             
        end

        file4 = fopen('outputt.out','w+'); 
        for k=1:numel(y)                                                                       
        fprintf(file4,' %4.6f \n',y(k) );             
        end
        cd(oldfolder)
        
    case 5
        cd('data/FODS_2')
        load inputtestactive.out;
        load outputtestactive.out;
        X=inputtestactive;
        y=outputtestactive;
        cd(oldfolder)
    case 6
        cd('data/FODS_3')
        load inpiecewise.out;
        load outpiecewise.out;

        X=inpiecewise;
        y=outpiecewise;
        cd(oldfolder)
    case 7
        cd('data/FODS_4')
        load inputt.out;

        X=inputt(:,1);
        y=inputt(:,2);
        cd(oldfolder)
    case 8
        cd('data/FODS_5')
        load inpiecewise2.out;
        load outpiecewise2.out;

        X=reshape(inpiecewise2,[],2);
        y=outpiecewise2;
        cd(oldfolder);
    case 9
        cd('data/FODS_6')
        load inputt.out;
        load outputt.out;

        X=reshape(inputt,[],2);
        y=outputt;
        cd(oldfolder)
    case 10
        cd('data/FODS_7')
        load chi_itg.dat;
        X=chi_itg(:,1:10);
        y=chi_itg(:,11);
        cd(oldfolder)   
    case 11

      cd('data/Tauth')
      load JM_tauth_data;

      X=[a0,b0,delta,ip,kappa,nebar,ploss,r0,zeff];
      y=[tauth];
      cd(oldfolder)
    case 12
        X=linspace(-10,10,1000);
        a=X;
        y=sinc(a);
        sdd=(0.05+0.21*(1+sin(2.*a)))./(1+exp(-0.2.*a));
        error=normrnd(0,sdd,1,1000);
        y=y+error;
        X=X';
        y=y';
      case 13
        cd('data/Mixture_model_1')
        load inn1.out;
        load out1.out;

        X=inn1;
        y=out1;
        cd(oldfolder);  
      case 14
        cd('data/Mixture_model_2')
        load inn2.out;
        load out2.out;

        X=inn2;
        y=out2;
        cd(oldfolder);   
    otherwise
            
        error('Data not specified correctly, Please select 1-14');

end

%% Summary of Data
file55 = fopen('Data_Summary.out','w+');  

    
    [a,b]=size(X);
    c=size(y,1);

   fprintf(file55,'The number of datapoints of inputs is : %d \n',a);
   fprintf(file55,'The number of features of inputs is : %d \n',b);
   fprintf(file55,'The number of datapoints of outputs is : %d \n',c);
%% Options for Training
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR-MM')
disp('2:MM-MM')
method=input('Enter the learning scheme desired: ');
disp('*******************************************************************')
%% Option to select the inducing points
disp('SELECT OPTION FOR INITIALISING INDUCING POINTS')
disp('1:K-means') % This throws an error sometimes
disp('2:Random')

method2=input('Enter the options for initialising inducing points: ');
if method2 > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Rescale data and then Split dataset to Train and Test;
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.1;
disp('')
if size(X,1)>=500
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
else
X_train=X;
y_train=y;
X_test=X;
y_test=y;
ind_train=1:size(X,1);
ind_test=1:size(X,1);
end
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
 if Datause==1
    Experts=6;
elseif Datause ==2
    Experts=7;
elseif Datause ==3
Experts=3;
elseif Datause ==4
Experts=2; 
    elseif Datause ==5
    Experts=4;
     elseif Datause ==6
    Experts=3; 
    elseif Datause ==7
    Experts=3;  
    elseif Datause ==8
    Experts=13; 
    elseif Datause ==9
    Experts=6; 
    elseif Datause ==10
    Experts=8; 
    elseif Datause ==11
    Experts=9; 
    elseif Datause ==12
    Experts=4;
    elseif Datause ==13
    Experts=3;     
else
    Experts=3;
 end    
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
Experts=input('Enter the maximum number of experts required: ');
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') % This throws an error sometimes
disp('2:User prefered options')

choicee=input('Enter the options for setting the NN classifier parameters: ');
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
disp('*******************************************************************')    
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Claasifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);

%% Choices used in the paper

end
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('CCR-MM SCHEME')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(Data,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
[modelNN] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ij,1}=hyper_updated;
    Xtrains{ij,1}=Xuse;
    ytrains{ij,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = predictNN(X_train, modelNN); % Predicts the Labels              
[Valuee1,~,cost3]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:2
fprintf('iteration %d | 2 ... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ik,1}=hyper_updated;
    Xtrains{ik,1}=Xuse;
    ytrains{ik,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels        
 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
%  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_CCR_MM';
mkdir(folder);
tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_tola,~] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,...
    X_test,y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform(hardtr,...
    softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer,stdclem];
headers = {'Hard_pred','Soft_pred','Stadev'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains)
cd(oldfolder)
else
disp('*******************************************************************')    
disp('---------------------------MM-MM SCHEME----------------------------') 
%  parpool('cluster1',8) 
tic;
 R2now=0; 
 meanfunc=[];
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
%% Starting MM loop
 for i=1:2
fprintf('iteration %d | 2... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
 disp('Initialised randomly for the first time')
else
weights=weights_updated;
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
disp('initialised using MM scheme')
end
diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{il,1}=hyper_updated;
    Xtrains{il,1}=Xuse;
    ytrains{il,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', il);     
end

if i==1
[Valueeini,~,costini]=prediction_clement(weights_updated,dd,X_train,...
    y_train,Xtrains,ytrains,Experts);
fprintf('R2 initial accuracy for random initialisation is %4.4f... .\n', costini.R2);   
end

if i==1
dd_updated=dd;
else
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
end

[modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
               hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %[dd_updated,D] = predictNN(X_train, modelNN); % Predicts the Labels 

 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;

    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
% [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions ); 
%%           
oldfolder=cd;
cd(oldfolder) % setting original directory
folder = 'Results_MM_MM';
mkdir(folder);
tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');
cd(folder)
saveas(gcf,'performance_a.fig')
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_tola,~] = predictNN(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_updated,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_tola,D] = predictNN(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_tola,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer,ind_train,ind_test,stdclem]=Plot_perform(hardtr,...
    softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

copyfile('Data_Summary.out',folder)
cd(folder)
file5 = fopen('Summary.out','w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer,stdclem];
headers = {'Hard_pred','Soft_pred','Stadev'}; 
csvwrite_with_headers('output_answer.csv',Matrix,headers);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
save('R2evolution.out','R2_allmm','-ascii')
save('L2evolution.out','L2_allmm','-ascii')
save('RMSEevolution.out','RMSE_allmm','-ascii')
save('Valueevolution.out','valueallmm','-ascii')
parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains)
cd(oldfolder) 

end
disp('*******************************************************************')
rmpath('CKS')
rmpath(mydir)
disp('---------------------PROGRAM EXECUTED SUCCESFULLY------------------')
 
end
disp('-----OVERALL PROGRAM EXECUTED SUCCESFULLY AND FILES SAVED-----------')     
