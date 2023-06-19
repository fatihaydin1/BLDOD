%% DEMO for BLDOD
clc;
clearvars -except synthetic X y;

tm = tic;
[score] = BLDOD(X, 1);
diff = toc(tm);
[outlier_factor, suspicious_index] = sort(score, 'descend');


[XROC, YROC, TROC, ROC] = perfcurve(y, score, 1);
plot(XROC, YROC);

[~,~,~,precision] = perfcurve(y, score, 1, 'XCrit', 'prec');
[~,~,~,recall] = perfcurve(y, score, 1, 'XCrit', 'reca');
Fmeasure = (2*precision*recall) / (precision+recall);



