function [hyper_updated]=optimise_experts(X_train,y_train)

nin = size(X_train,2);			% input Layer.

mlp_nhidden = 100;

dim_target = 1;			% Dimension of target space

				% Make variance small for good starting point
options = foptions;
options(1) = 1;			% This provides display of error values.
options(14) = 3000; 

net2 = mlp(nin, mlp_nhidden, dim_target, 'linear');

[hyper_updated, options] = netopt(net2, options, X_train, y_train, 'quasinew');


end