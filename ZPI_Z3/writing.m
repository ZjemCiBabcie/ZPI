load('G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\siedzenie_przodem.mat');   
% Acceleration(:, 1:end-1)
% tt = synchronize(Acceleration(:, :-1), Orientation(:, 2:3), AngularVelocity(:, 2:3), Position, 'regular','linear','TimeStep',seconds(1))
tt = synchronize(Acceleration(:, 1:end-1),'regular', 'linear', 'TimeStep',seconds(1))
writetimetable(tt, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\siedzenie_przodem.csv')
