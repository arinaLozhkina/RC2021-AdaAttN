close all
clear

load ALLLOSSES

%% Online examples

% MSE_AdaAttN(MSE_AdaAttN > 0.2) = 0.2;
% MSE_AdaIn(MSE_AdaIn > 0.2) = 0.2;
% MSE_SANnet(MSE_SANnet > 0.2) = 0.2;

DataX{3} = loss_AdaAttN; % Create dataset of 3 groups
DataX{2} = loss_AdaIn;
DataX{1} = loss_SAN;

x = 1:1:20;
Labels = [];

for i = 1:numel(x)
    Labels{i} = ['Style ', num2str(x(i))]; % Create labels for each column
end

Groups = {' SanNet', ' AdaIn', ' AdaAttN'}; % Create labels for each group

[Fig] = CirHeatmap(DataX', 'Colormap', 'jet' ,'GroupLabels',  Groups,'OuterLabels', Labels, 'CircType', 'tq','InnerSpacerSize',0.1);
