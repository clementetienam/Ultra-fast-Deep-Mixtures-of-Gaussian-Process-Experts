function parsave(weights_updated,datass)
save('Regressor.mat', 'weights_updated');
save('Dataused.out','datass','-ascii')
end