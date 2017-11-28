%% This code is used to read in data and try spectral clustering.
% result of the clustering is saved in mat file.
% 1. Construct graph using original edges, try spectral clustering with
% unnormalized laplacian
% 2. Construct graph using simlarity matrix from sentiment analysis, 
% try spectral clustering with unnormalized laplacian
% 3. Construct graph using original edges, try spectral clustering with
% normalized symmetric laplacian
% 4. Construct graph using simlarity matrix from sentiment analysis, 
% try spectral clustering with normalized symmetric laplacian

clear all;
fileName = './FinalData/BernieSandersEdgesFrequencyReducedSymmetricNewLabel.csv';
BS = csvread(fileName);
fromNode = BS(:,1);
toNode = BS(:,2);
weights = BS(:,3);
%% 1. Create Graph with unnormalized laplacian
G = graph(fromNode,toNode,weights);
W = G.adjacency;
d = sum(W);
D = diag(d);
L = D - W;

K = 2;
[EVector,EValue,flag] = eigs(L,K);

V = EVector(:,1:K);
idx = kmeans(V,K);
X = V(1:1000,1);
Y = V(1:1000,2);
scatter(X,Y);
tabulate(idx);
%nnz(idx)
%% 2. Construct similarity matrix using sentiment analysis, try unnomralized Laplacian
clear all;
fileName = './FinalData/HillaryClintonUserSentimentNewLabel.csv';
BS = csvread(fileName);
BSSorted = sortrows(BS);
Index = [2,4];
BSFeatures = BSSorted(:,Index);
W = squareform(pdist(BSFeatures));
d = sum(W);
D = diag(d);
L = D - W;

K = 5;
[EVector,EValue,flag] = eigs(L,K);
V = EVector(:,1:K);
idx = kmeans(V,K);
X = V(1:1000,1);
Y = V(1:1000,2);
scatter(X,Y);
tabulate(idx);

%% 3. Normalized Laplacian Method with Original Edges
clear all;
fileName = './FinalData/DonaldTrumpEdgesFrequencyReducedSymmetricNewLabel.csv';
BS = csvread(fileName);
fromNode = BS(:,1);
toNode = BS(:,2);
weights = BS(:,3);
G = graph(fromNode,toNode,weights);
W = full(G.adjacency);
d = sum(W);
D = diag(d);
DPrime = D^(-1/2);

%% One can change k values to change the number of clusters
n = size(DPrime,1);
Lsym = eye(n) - DPrime * W * DPrime;

K = 2;
[EVector,EValue,flag] = eigs(Lsym,2);
V = EVector(:,1:K);
U = normr(V);
idx = kmeans(U,K);
X = U(1:1000,1);
Y = U(1:1000,2);
scatter(X,Y);
tabulate(idx);

%% 4. Normalized Laplacian Method with Sentiment
clear all;
fileName = './FinalData/HillaryClintonUserSentimentNewLabel.csv';
BS = csvread(fileName);
BSSorted = sortrows(BS);
Index = [2,4];
BSFeatures = BSSorted(:,Index);
W = squareform(pdist(BSFeatures));
d = sum(W);
D = diag(d);
DPrime = D^(-1/2);


%% One can change k values to change the number of clusters
n = size(DPrime,1);
Lsym = eye(n) - DPrime * W * DPrime;

K = 5;
[EVector,EValue,flag] = eigs(Lsym,K);
V = EVector(:,1:K);
U = normr(V);
idx = kmeans(U,K);
X = U(1:1000,1);
Y = U(1:1000,2);
scatter(X,Y);
tabulate(idx);