load('G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\opolska-brzezina.mat');

writetimetable(Acceleration, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\opolska-brzezina.xlsx', 'Sheet', 'Acceleration');
writetimetable(AngularVelocity, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\opolska-brzezina.xlsx',  'Sheet', 'AngularVelocity');
writetimetable(Orientation, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\opolska-brzezina.xlsx',  'Sheet', 'Orientation');
writetimetable(Position, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\opolska-brzezina.xlsx',  'Sheet', 'Position');    