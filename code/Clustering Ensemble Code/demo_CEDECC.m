% This is a demo for the CEDECC clustering ensemble algorithms.      %
% If you find this code useful for your research, please cite the   %
% paper after de-anonymised.                                        %

function [pred_Result, time] = demo_CEDECC(argument)

dataName = argument.dataName;
M = argument.M;
gt = [];
load([dataName,'.mat']);
tol_cc = cluster_confidence_Before + cluster_confidence_After;
[~,i] = sort(-tol_cc(1,:));
rank_cluster = cluster_result(:,i);

if min(gt) == 0
    gt = gt + 1;
end
[N, tol_M] = size(rank_cluster);
para_theta = 0.4;
cntTimes = argument.cntTimes; 
Mm = 10 + round(rand(1,10)*(M-10));

clsNums = numel(unique(gt));

for runIdx = 1:cntTimes

    M = Mm(runIdx);
    baseCls = rank_cluster(:,1:M);
    baseCls = baseCls+1;
    
    t1 = clock;
    
    %% Get all clusters in the ensemble
    [bcs, baseClsSegs] = getAllSegs(baseCls);
    
    %% Compute ECE
    ECE = computeECE(bcs, baseClsSegs, para_theta);
    resultsCEDECC = CE(bcs, baseClsSegs, ECE, clsNums);     
    t2 = clock;
    if min(resultsCEDECC) == 0
        resultsCEDECC = resultsCEDECC + 1;
    end
    
    pred_CEDECC(:,runIdx) = resultsCEDECC;
    CEDECC_time(runIdx) = etime(t2,t1);  
end
CEDECC_time = mean(CEDECC_time);
pred_Result = {pred_CEDECC};
time = CEDECC_time;
