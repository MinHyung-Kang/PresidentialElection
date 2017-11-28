
%% Uses the result of spectral clustering to classify the users and look at the graph.
% 1. Load the data, results of spectral clustering and index computation
% 2. Get graphs for each cluster (not used for plotting all graph)
% 3. Plot the graph
% 4. Look at the greatest component

clear all;
% Load the data
fileName = './FinalData/DonaldTrumpUserSentimentNewLabel.csv';
Sentiment = csvread(fileName);

fileName = './FinalData/DonaldTrumpEdgesFrequencyReducedSymmetricNewLabel.csv';
BS = csvread(fileName);
fromNode = BS(:,1);
toNode = BS(:,2);
weights = BS(:,3);

%%
% Load index computation
load trump_sentimentedges_k5.mat
fileName = './FinalData/DonaldTrumpClusteringClass.csv';
Class = csvread(fileName);
classes = Class(:,2);

tabulate(idx)
k=5;
Group = cell(1,k);
GroupIndices = cell(1,k);
for i = 1:k
    tempIndex = 1:length(idx);
    GroupIndices{i} = tempIndex(idx == i);
    Group{i} = Sentiment(idx == i, 2:4);
    disp(mean(Group{i}));
end

%% 2. Get graphs within each cluster
clusteredGraphs = cell(1,k); 

from = ismember(fromNode, GroupIndices{1});
to = ismember(toNode, GroupIndices{1});
combined = (from & to);
for i = 1:k
    from = ismember(fromNode, GroupIndices{i});
    to = ismember(toNode, GroupIndices{i});
    combined = (from & to);
    clusteredGraphs{i} = BS(combined,1:3);
end
%% Constants to be used for drawing the graph
HillaryMap = [0,0,0
              0,1,0
              1,0,0
              0,1,0
              0,1,0
              0,0,1]; % 0,Neu, Neg, Neu, Neu, Pos
HillaryLabel = {'Unknown(0)','Neutral(1)','Negative(2)','Neutral(3)','Netural(4)','Positive(5)'};

SandersMap = [0,0,0
              0,1,0
              0,1,0
              0,0,1
              1,0,0
              0,1,0]; % 0,Neu, Neu, Pos, Neg, Neu
SandersLabel = {'Unknown(0)','Neutral(1)','Neutral(2)','Positive(3)','Negative(/Neutral)(4)','Neutral(5)'};

TrumpMap = [0,0,0
              1,0,0
              0,1,0
              0,1,0
              0,1,0
              0,0,1]; % 0,Neg, Neu, Neu, Neu, Pos
TrumpLabel = {'Unknown(0)','Negative(1)','Neutral(2)','Neutral(3)','Netural(4)','Positive(5)'};
%% 3. Plot graph

% Change plot size
combinedGraph = graph(fromNode, toNode, weights);

combinedGraph.Nodes.Class = classes;
hFig = figure(1);
width = 1000;
height = 1000;
set(hFig, 'Position',[0 0 width height]);

% Change plot settings
%p = plot(combinedGraph,'Layout','force');
%p = plot(combinedGraph,'Layout','layered');
p = plot(combinedGraph);
p.NodeCData = classes;
p.MarkerSize = 4;
p.LineWidth = log(weights);

cb = colorbar('south');
set(gca,'Position',[.05 .05 .9 .9])
set(cb,'YTick',[0.4,1.25,2.1,2.9,3.75,4.55]);
colormap(TrumpMap)
set(cb,'YTickLabel',TrumpLabel);

% Add other information
title('Donald Trump Network')
axis off;


%% 4. Look at the giant Component
nn = numnodes(combinedGraph);
[s,t] = findedge(combinedGraph);
weightedAdj = sparse(s,t,combinedGraph.Edges.Weight,nn,nn);
[GC, I] = giantComponent(weightedAdj);

save('trump_gcc','GC','I','weightedAdj')


%%
GCC = graph(GCSym);
hFig = figure(1);
width = 1000;
height = 1000;
set(hFig, 'Position',[0 0 width height]);

% Change plot settings
%q = plot(GCC);
q = plot(GCC,'Layout','force');
%q = plot(GCC,'Layout','layered');
q.NodeCData = classes(I);
q.MarkerSize = 4;
%q.LineWidth = log();

cb = colorbar('south');
set(gca,'Position',[.05 .05 .9 .9])
set(cb,'YTick',[0.4,1.25,2.1,2.9,3.75,4.55]);
colormap(TrumpMap)
set(cb,'YTickLabel',TrumpLabel);

% Add other information
title('Donald Trump Network Greatest Component')
axis off;



%% Look at the greatest connected component Statistics
numNode = numNodes(GCSym);
numEdge = numEdges(GCSym);
edgeDensity = linkDensity(GCSym);
numLeaf = length(leafNodes(GCSym));
numTriangle =  loops3(GCSym);
globalCC = transitivity(GCSym);
localCC = clustCoeff(GCSym);
diameter = diameter(GCSym);
radius = graphRadius(GCSym);
avPathLength = avePathLength(GCsym);