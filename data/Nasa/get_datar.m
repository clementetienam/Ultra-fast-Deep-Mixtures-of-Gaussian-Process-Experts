function [X,y]= get_datar()
load ('nasadata.txt');
aa=nasadata;
X=aa(:,1:3);
y=nasadata(:,4);
end