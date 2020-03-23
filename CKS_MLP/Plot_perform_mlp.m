function [Valueehard]= Plot_perform_mlp(hardtr,hardts,y,folder,X,ind_train,ind_test,oldfolder,Datause)

Valueehard=zeros(size(X,1),1);
Valueehard(ind_train,:)=hardtr;
Valueehard(ind_test,:)=hardts;

if size(X,2)==1   
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


cd(folder)
saveas(gcf,'performance.fig')
cd(oldfolder)
end

if (size(X,2)>=2)
    
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


cd(folder)
saveas(gcf,'performance.fig')
cd(oldfolder)
end

end