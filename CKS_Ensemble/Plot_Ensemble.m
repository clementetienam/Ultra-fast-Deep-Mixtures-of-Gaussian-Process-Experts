function[clem1,clem1std,clem2,clem2std]=  Plot_Ensemble(folder,oldfolder,predict_hard,predict_soft,iterra,y)
N=iterra;
b=(1:size(y,1))';

linecolor1=colordg(4);
clem1=mean(predict_hard,2);
clem1std=std(predict_hard,0,2);

clem2=mean(predict_soft,2);
clem2std=std(predict_soft,0,2);


figure()
plot(y,predict_hard(:,1:N),'+','Color',linecolor1,'LineWidth',2)
 xlabel('True','FontName','Helvetica', 'Fontsize', 13);
ylabel('Machine','FontName','Helvetica', 'Fontsize', 13);
title('Ensemble evolution of Machine','FontName','Helvetica', 'Fontsize', 13)
aa = get(gca,'Children');
hold on
plot(y,clem2,'*b','LineWidth',1)
bb2 = get(gca,'Children');
hold on
plot(y,clem1,'ok','LineWidth',1)
bb1 = get(gca,'Children');
hold on
plot(y,y,'r','LineWidth',1)
bb = get(gca,'Children');

 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [bb;bb1;bb2;aa];
legend(h,'Ground Truth','Mean Hard prediction','Mean Soft prediction','Realisations','location','northeast');
hold off
cd(folder)
saveas(gcf,'performance_ensemble.fig')
cd(oldfolder)
end