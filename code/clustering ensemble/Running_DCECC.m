% This is a demo for the DCECC clustering ensemble algorithms.      %
% If you find this code useful for your research, please cite the   %
% paper after de-anonymised.  

clc;
addpath(genpath(pwd))
dataName = 'cars_result'; 
M = 20; % Ensemble size
runtime = 20; % How many times will be run.
argument.DCECC.dataName = dataName; 
argument.DCECC.M = M; 
argument.DCECC.cntTimes = runtime;
DCECC_argument = argument.DCECC;
[DCECC.pred, DCECC.time] = demo_DCECC(DCECC_argument);

clear dataName DCECC_argument M runtime argument;

