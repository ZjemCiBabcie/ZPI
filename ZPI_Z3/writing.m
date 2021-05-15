load('G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\kierowcaA.mat');

writetimetable(Acceleration, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\kierowcaA.xlsx', 'Sheet', 'Acceleration');
writetimetable(AngularVelocity, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\kierowcaA.xlsx',  'Sheet', 'AngularVelocity');
writetimetable(Orientation, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\kierowcaA.xlsx',  'Sheet', 'Orientation');
writetimetable(Position, 'G:\My Drive\PWR\ZPI\ZPI\ZPI_Z3\kierowcaA.xlsx',  'Sheet', 'Position');    