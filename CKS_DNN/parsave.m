function parsave(weights_updated,modelNN,Class_all,clfy)
save('Regressor.mat', 'weights_updated');
save('Classifier.mat', 'modelNN');
save('clfy.mat', 'clfy');
save('Class_all.mat', 'Class_all');
end