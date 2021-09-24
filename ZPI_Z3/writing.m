clc
clear
load('example2.mat')
acc = synchronize(Acceleration(:, 2:end-1),'regular', 'linear', 'TimeStep',seconds(1));
speed = Position(:, 4);
writetimetable(acc, 'acceleration.csv')
writetimetable(speed, 'speed.csv')