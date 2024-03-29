function labels = MM_clement(weights,X,y,modelNN,Class_all,Experts)
% weights=weights_updated;
	meanfunc=[];
 likfunc = {@likGauss};    

  inf = @infGaussLik;
%  inf=@infGaussLik;
 cov = {@covSEiso}; 
 infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
 dd=size(X,1);
 m=zeros(size(X,1),1);
 outputtR=zeros(dd,Experts);
 outputS=zeros(dd,Experts);
for L=1:Experts
    Classuse=Class_all{L,:};
    if size(X(Classuse),1)>= 2
    weigt=weights{L,:};
    
	cov1 = {'apxSparse',cov,weigt.xu};     
    [m s2] = gp(weigt, infv, meanfunc, cov1, likfunc, X(Classuse,:), y(Classuse,:), X);
  
     outputtR(:,L)=m;
     outputS(:,L)=s2;
    else
    outputtR(:,L)=zeros(size(X,1),1);
     outputS(:,L)=zeros(size(X,1),1);
    end


end

%% softmax
[p,D] = predictNN(X, modelNN); 


% First_term=-(log(D));
% for i=1:5
% second_term(:,i)= 0.5*log((outputS(:,i)));
% end
% 
% for i=1:Experts
%  under=1/(2.*outputS(:,i));
% third_term(:,i)= under*((y-outputtR(:,i)).^2);
% end
% 
% Alll= First_term+second_term+third_term;

for i=1:Experts

thirds_term(:,i)= ((y-outputtR(:,i)).^2);

end

for i=1:size(X,1)
[clem,clem2]=min(thirds_term(i,:));
clemall(:,i)=clem2;
end
labels=clemall';
end