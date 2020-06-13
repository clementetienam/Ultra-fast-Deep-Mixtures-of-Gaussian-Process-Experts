function parsave(hardanswer,softanswer,weights_updated,modelNN,Class_all,clfy,datass)
save('predict_hard.out','hardanswer','-ascii')
save('predict_soft.out','softanswer','-ascii')
save('GP.mat', 'weights_updated');
save('Classifier.mat', 'modelNN');
save('clfy.mat', 'clfy');
save('Class_all.mat', 'Class_all');
save('Dataused.out','datass','-ascii')
end