%% DEMO for BLDOD
clc;
clearvars;

% Load a data set from the "outlier datasets" folder (variables X and y)
load(fullfile('..', 'outlier datasets', 'wine.mat'));

tm = tic;
score = BLDOD(X, 1);
elapsedTime = toc(tm);
[outlier_factor, suspicious_index] = sort(score, 'descend');

% ROC curve and the area under it
[XROC, YROC, ~, ROC] = perfcurve(y, score, 1);
plot(XROC, YROC);

% Precision-recall curve and the area under it
[~, ~, ~, PRAUC] = perfcurve(y, score, 1, 'XCrit', 'reca', 'YCrit', 'prec');

% F-measure when points whose factor is above 0.9 are labelled outliers
predicted = score > 0.9;
TP = sum(predicted & (y == 1));
precision = TP / max(sum(predicted), 1);
recall = TP / sum(y == 1);
Fmeasure = 2 * precision * recall / max(precision + recall, eps);
