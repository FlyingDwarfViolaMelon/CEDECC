% This is a demo for the CEDECC clustering ensemble algorithms.      %
% If you find this code useful for your research, please cite the   %
% paper after de-anonymised.                                        %

function labels = CE(bcs,baseClsSegs, ECI, clsNum)

% Build the weighted graph
lwB = bsxfun(@times, baseClsSegs, ECI);

% Partition the graph
labels = zeros(size(bcs,1),numel(clsNum));
for i = 1:numel(clsNum)  
    labels(:,i) = bipartiteGraphPartitioning(lwB',clsNum(i));
end 

function labels = bipartiteGraphPartitioning(B,Nseg)

% B - |X|-by-|Y|, cross-affinity-matrix

[Nx,Ny] = size(B);
if Ny < Nseg
    error('The cluster number is too large!');
end

dx = sum(B,2);
dx(dx==0) = 1e-10;
Dx = sparse(1:Nx,1:Nx,1./dx); clear dx
Wy = B'*Dx*B;

% compute Ncut eigenvectors
% normalized affinity matrix
d = sum(Wy,2);
D = sparse(1:Ny,1:Ny,1./sqrt(d)); clear d
nWy = D*Wy*D; clear Wy
nWy = (nWy+nWy')/2;

% computer eigenvectors
[evec,eval] = eig(full(nWy)); clear nWy
[~,idx] = sort(diag(eval),'descend');
Ncut_evec = D*evec(:,idx(1:Nseg)); clear D

% compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
evec = Dx * B * Ncut_evec; clear B Dx Ncut_evec

% normalize each row to unit norm
evec = bsxfun( @rdivide, evec, sqrt(sum(evec.*evec,2)) + 1e-10 );

% k-means
labels = kmeans(evec,Nseg,'MaxIter',100,'Replicates',3);
