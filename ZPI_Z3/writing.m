load('G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\training-set.mat')
tt = synchronize(Acceleration(:, 1:end-1),'regular', 'linear', 'TimeStep',seconds(1))
writetimetable(tt, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\training-set.csv')

load('G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\test-set.mat')
tt = synchronize(Acceleration(:, 1:end-1),'regular', 'linear', 'TimeStep',seconds(1))
writetimetable(tt, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\test-set.csv')
