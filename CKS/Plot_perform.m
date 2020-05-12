function [Valueehard,Valueesoft,ind_train,ind_test,stdclem]= Plot_perform(hardtr,softtr,hardts...
,softts,y,method,folder,X,ind_train,ind_test,oldfolder,Datause,stdtr,stdte)

Valueehard=zeros(size(X,1),1);
Valueehard(ind_train,:)=hardtr;
Valueehard(ind_test,:)=hardts;



Valueesoft=zeros(size(X,1),1);
Valueesoft(ind_train,:)=softtr;
Valueesoft(ind_test,:)=softts;

stdclem=zeros(size(X,1),1);
stdclem(ind_train,:)=stdtr;
stdclem(ind_test,:)=stdte;

%if method= 1 || &&
if (method==1 || 2 || 3 )  && size(X,2)==1   
figure()
subplot(2,3,1)
plot(X,y,'+r');
hold on
plot(X,Valueehard,'.k')
shading flat
grid off
title('(a)-Machine Reconstruction(Hard-Prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
h = legend('True y','Machine');set(h,'FontSize',10);

subplot(2,3,2)
scatter(Valueehard,y,'o');
shading flat
grid off
title('(b)-Machine Reconstruction(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Machine', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('True', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,3)
hist(Valueehard-y)
shading flat
grid off
title('(c)-Dissimilarity(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,4)
plot(X,y,'+r');
hold on
plot(X,Valueesoft,'.k')
shading flat
grid off
title('(d)-Machine Reconstruction(Soft-Prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
h = legend('True y','Machine');set(h,'FontSize',10);

subplot(2,3,5)
scatter(Valueesoft,y,'o');
shading flat
grid off
title('(e)-Machine Reconstruction(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Machine', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('True', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,6)
hist(Valueesoft-y)
shading flat
grid off
title('(f)-Dissimilarity(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
cd(folder)
saveas(gcf,'performance.fig')
cd(oldfolder)
end

if (method==1 || 2 || 3 )  && (size(X,2)>=2)
    
figure()
subplot(2,2,1)
scatter(Valueehard,y,'o');
shading flat
grid off
title('(a)-Machine Reconstruction(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Machine', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('True', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,2)
hist(Valueehard-y)
shading flat
grid off
title('(b)-Dissimilarity(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,3)
scatter(Valueesoft,y,'o');
shading flat
grid off
title('(c)-Machine Reconstruction(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Machine', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('True', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,4)
hist(Valueesoft-y)
shading flat
grid off
title('(d)-Dissimilarity(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
cd(folder)
saveas(gcf,'performance.fig')
cd(oldfolder)
end

end