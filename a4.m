function a4_20292366
% Function for CISC271, Winter 2022, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);

%     for idx = 1:17
%         jx = idx;
%         jy = idx + 1;
%         Xx = dmrisk(:, (1:size(dmrisk,2))~=jx);
%         Xy = dmrisk(:, (1:size(dmrisk,2))~=jy);
%         yx = dmrisk(:,jx);
%         yx = dmrisk(:,jy);
%         [~,rx] = pca(zscore(Xx), 'NumComponents', 2);
%         [~,ry] = pca(zscore(Xy), 'NumComponents', 2);
%         
%     end
    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 17;
    jOB = 16;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM, zDM, qOB, zOB] = a4q1(rDM, yDM, rOB, yOB);

    % Plots the relevant data for later interpretation in the discussions
    % section
    
    % Shows the LDA axis for the Diabetes data points
    figure;
    gscatter(zDM, yDM, yDM, 'rb');
    xlabel('LDA Score'); ylabel('Class');
    ylim([-2 2]);
    title('LDA Scores for Diabetes Data');
    
    % Shows the LDA axis for the Obesity data points
    figure;
    gscatter(zOB, yOB, yOB, 'rb');
    xlabel('LDA Score'); ylabel('Class');
    ylim([-2 2]);
    title('LDA Scores for Obesity Data');
    
    % Shows the PCA Scatterplot for Diabetes data points
    figure;
    scatter(rDM(:,1), rDM(:,2), [], yDM, 'filled');
    xlabel('PCA 1'); ylabel('PCA 2');
    title('PCA Scatterplot for Diabetes Data');
    
    % Shows the PCA Scatterplot for Obesity data points
    figure;
    scatter(rOB(:,1), rOB(:,2), [], yOB, 'filled');
    xlabel('PCA 1'); ylabel('PCA 2');
    title('PCA Scatterplot for Obesity Data');

    % Compute the ROC curve and its AUC using the roccurve function
    [fprDM, tprDM, aucDM, boptDM] = roccurve(yDM, zDM);
    [fprOB, tprOB, aucOB, boptOB] = roccurve(yOB, zOB);

    % Plot ROC curves using tpr and fpr calculated in roccurve function
    % using the confmat function
    figure;
    plot(fprDM, tprDM, 'b-', 'LineWidth', 2); hold on;
    plot(fprOB, tprOB, 'r-', 'LineWidth', 2);
    xlim([0 1]); ylim([0 1]);
    plot([0 1], [0 1], 'k--', 'LineWidth', 1.5); hold off;
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('ROC Curves for Diabetes and Obesity Data');
    legend(sprintf('Diabetes (AUC = %.3f)', aucDM), sprintf('Obesity (AUC = %.3f)', aucOB));
    
    % Compute the confusion matrices for diabetes and obesity at their
    % optimal threshold
    cmatDM = confmat(yDM, zDM, boptDM);
    cmatOB = confmat(yOB, zOB, boptOB);

    % Display the confusion matrices
    disp("Confusion matrix for diabetes:");
    disp(cmatDM);
    disp("Confusion matrix for obesity:");
    disp(cmatOB);
    disp("bopt value for diabetes:")
    disp(boptDM)
    disp("bopt value for obesity:")
    disp(boptOB)
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    % Creates a 'classifier' using LDA using labels from yvecs and data from
    % Xmats - Here, labels of 1 are in the positive class and -1 in the
    % negative class
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % Projects the LDA classifiers into a lower dimensional space for PCA
    % reduction effectiveness
    z1 = Xmat1 * q1;
    z2 = Xmat2 * q2;
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation
    
    % Initializes qvec into a Nx1 vector of ones and calculates the means
    % of the data for accurate classification and LDA axis calculates
    qvec = ones(size(X1,2), 1);
    xbar1 = mean(X1);
    xbar2 = mean(X2);

    % Compute the within-class means and scatter matrices
    M1 = size(X1, 1);
    M2 = size(X2, 1);
    S1 = (X1 - ones(M1,1)*xbar1)'*(X1 - ones(M1,1)*xbar1);
    S2 = (X2 - ones(M2,1)*xbar2)'*(X2 - ones(M2,1)*xbar2);
    Sw = S1 + S2;

    % Compute the between-class scatter matrix

    Sb = (xbar1 - xbar2)'*(xbar1 - xbar2);

    % Fisher's linear discriminant is the largest eigenvector
    % of the Rayleigh quotient

    % Spectral decomposition is used to create a binary separation
    [V, D] = eig(inv(Sw)*Sb);
    qvec= V(:,1);

    % May need to correct the sign of qvec to point towards mean of X1
    if (xbar1 - xbar2)*qvec < 0
        qvec = -qvec;
    end
end

function [fpr tpr auc bopt] = roccurve(yvec_in,zvec_in)
% [FPR TPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
% ROC curve and related values for labels YVEC and scores ZVEC.
% Unique scores are used as thresholds for binary classification.
%
% INPUTS:
%         YVEC - Mx1 labels, +/- computed as ==/~= 1
%         ZVEC - Mx1 scores, real numbers
% OUTPUTS:
%         FPR  - Kx1 vector of False Positive Rate values
%         TPR  - Kx1 vector of  True Positive Rate values
%         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
%         BOPT - scalar, optimal threshold for accuracy

% Sort the scores and permute the labels accordingly
[zvec, zndx] = sort(zvec_in);
yvec = yvec_in(zndx);

% Sort and find a unique subset of the scores; problem size
bvec = unique(zvec);
bm = numel(bvec);

% Compute a confusion matrix for each unique threshold value;
% extract normalized entries into TPR and FPR vectors; track
% the accuracy and optimal B threshold
tpr = [];
fpr = [];
acc = -inf;
bopt = -inf;
for jx = 1:bm
    % Compute confusion matrix for each unique threshold value
    cmat = confmat(yvec, zvec, bvec(jx));
    tpr(jx) = cmat(1,1) / (cmat(1,1) + cmat(1,2));
    fpr(jx) = cmat(2,1) / (cmat(2,1) + cmat(2,2));

    % Finds the current accuracy and bopt
    curracc = (cmat(1,1) + cmat(2,2)) / (sum(cmat(1,:)) + sum(cmat(2,:)));
    if curracc > acc
        % Highest cmat accuracy = bopt
        acc = curracc;
        bopt = bvec(jx);
    end
end
% Compute AUC for this ROC
auc = aucofroc(fpr, tpr);
end

function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

    % Find the plus/minus 1 vector of quantizations
    qvec = sign((zvec >= theta) - 0.5);
    TP = sum((yvec == 1) & (qvec == 1));
    FN = sum((yvec == 1) & (qvec == -1));
    FP = sum((yvec == -1) & (qvec == 1));
    TN = sum((yvec == -1) & (qvec == -1));
    % Compute the confusion matrix by entries
    cmat = [TP FN; FP TN];
end

function auc = aucofroc(fpr, tpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X undx] = sort(reshape(fpr, 1, numel(fpr)));
    Y = sort(reshape(tpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 X 1] , [0 Y 1]));
end