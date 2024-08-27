% This is a demo for the CEDECC clustering ensemble algorithms.      %
% If you find this code useful for your research, please cite the   %
% paper after de-anonymised.  

clc;
addpath(genpath(pwd))
dataName = 'cars_result'; 
M = 20; % Ensemble size
runtime = 20; % How many times will be run.
argument.CEDECC.dataName = dataName; 
argument.CEDECC.M = M; 
argument.CEDECC.cntTimes = runtime;
CEDECC_argument = argument.CEDECC;
[CEDECC.pred, CEDECC.time] = demo_CEDECC(CEDECC_argument);

clear dataName CEDECC_argument M runtime argument;

