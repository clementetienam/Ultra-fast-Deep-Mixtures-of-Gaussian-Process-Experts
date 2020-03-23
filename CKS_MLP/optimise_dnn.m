function [hyper_updated]=optimise_dnn(X_train,y_train)
Xuse=X_train;
yuse=y_train;
input_count = size( Xuse , 2 );
output_count = size( yuse , 2 );

layers = [ ...
    sequenceInputLayer(input_count)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(40)
    reluLayer
    fullyConnectedLayer(output_count)
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',2000, ...
    'MiniBatchSize', 5 , ...
    'ValidationFrequency',10, ...
    'ValidationPatience',5, ...
    'Verbose',true, ...
    'Plots','training-progress');


hyper_updated = trainNetwork(Xuse',yuse',layers,options);


end