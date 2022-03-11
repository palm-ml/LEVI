function model = amsvr(X, Y, para)
% AMSVR     Adapted Multi-output SVR
%
% Description
%   [BETA, B, SVINDEX] = AMSVR(X, Y, PARA) means Adapted Multi-output SVR
%   
%   Statement
%   The function model = amsvr(X, Y, para) here is adapted
%   from the original algorithm msvr writed by Fernando Pérez Cruz. We modify it 
%   to fit the ML^2 condition. The main difference between is the addition of the
%   inconsistent penalty between the signs of the predicted values and given values.
%
%   Inputs,
%       X : data matrix with training samples in rows and features in in columns (N x D)
%       Y : numerical label matrix corresponding to the training samples in X above (N x L)
%       PARA : model parameters of the AMSVR model.
%
%   Outputs,
%       MODEL : trained model parameters
%
%   Extended description of input/ouput variables
%   PARA,
%       PARA.TOL :  tolerance during the iteration
%   	PARA.EPSI : instances whose distance computed is more than epsi should be penalized
%       PARA.C1 :   penalty parameter
%       PARA.C2 :   penalty parameter
%       PARA.KER :  type of kernel function ('lin', 'poly', 'rbf', 'sam')
%       PARA.PAR :  parameters of kernel function
%           SIGMA:  width of the RBF and sam kernel
%           BIAS:   bias in the linear and polinomial kernel
%           DEGREE: degree in the polynomial kernel
%   MODEL,
%       MODEL.BETA :    coeficient matrix of trainFeature's linear combination (N x L)
%       MODEL.B :       intercept matrix (1 x L)
%       MODEL.SVINDEX : support vectors' subscripts of row in trainFeature
% proccess the trainDistribution
N = size(X,1); %count of training samples
L = size(Y,2); %dimension of the label distribution
YS = sign(Y);  %sign of the label

% build the kernel matrix on the labeled samples (N x N)
H = kernelmatrix(para.ker, para.par, X, X);

% create matrix for regression parameters
Beta = ones(N, L);   
b = ones(1, L);

% E = prediction error per output (N x L)
P = H*Beta + repmat(b, N, 1);
E = Y - P;  
% compute the Euclidean distance of each examples
u = sqrt(sum(E.^2,2)); %u = RSE (N x 1)

% RMSE
RMSE(1) = sqrt(mean(u.^2));

% points for which prediction error is larger than epsilon, i.e., find SVs whose loss function != 0
i1 = find(u>=para.epsi);

% set initial values of alphas (N x 1)
a = 2*(u-para.epsi)./u;

% compute L1. we modify only entries for which  u > epsi.
L1 = zeros(size(u));
L1(i1) = u(i1).^2-2*para.epsi*u(i1)+para.epsi^2;   
% L2
L2 = YS .* P;
i2 = L2 >= 0;
L2(i2) = 0;
L2 = -sum(L2,2);

%Lp is the quantity to minimize (sq norm of parameters + slacks)
L_T(1)=sum(diag(Beta'*H*Beta))/2 + para.C1*sum(L1)/2 + para.C2*sum(L2);

%initial variables used in loopk
eta=1; %step length
k=1; %iteration number
hacer=1; %sentinel of loop
val=1; %sign of whether find support vectors

%strat training
while(hacer)
    
    % Print the iteration information.
    %fprintf('---> Iter:%4d, L_T:%15.7f, RMSE:%15.7f\n', k, L_T(k), RMSE(k));
    
    %next iteration
    k = k+1; 
    
    %save the model parameters in the previous step
    Beta_a = Beta;
    b_a = b;
    i1_a = i1;
	YS_a = YS;
    
    % M1 = [C1*K + D_a  C1] = y (only for obs i1)
    M11=[para.C1*H(i1,i1)+diag(1./a(i1))];
    M12 = para.C1*ones(size(M11,1),1);
    M21 = para.C1*a(i1)'*H(i1,i1);
    M22 = para.C1*sum(a(i1));
    M = [M11 M12; M21 M22];
    M=M+1e-11*eye(size(M,1));

    %compute betas
	YS_a(i2) = 0;
    M1 = para.C1*Y(i1,:) + para.C2*YS_a(i1,:)./repmat(a(i1), 1, L);
    M2 = para.C1*(a(i1)'*Y(i1,:))+para.C2*sum(YS_a(i1,:));
    sal1=inv(M)*[M1; M2];
    b_sal1=sal1(end,:);
    sal1=sal1(1:end-1,:);
    
    Beta = zeros(size(Beta));
    Beta(i1,:) = sal1;
    b = b_sal1;
	
    %recompute error
    P = H*Beta + repmat(b, N, 1);
    E = Y - P;
    %recompute i1 and u_z
    u=sqrt(sum(E.^2,2));
    i1=find(u>=para.epsi);
    
    %recompute loss function 
    L1 = zeros(size(u));
    L1(i1) = u(i1).^2-2*para.epsi*u(i1)+para.epsi^2;
    L2 = YS(i1,:) .* P(i1,:);
    i2 = L2 >= 0;
    L2(i2) = 0;
    L2 = -sum(L2,2);
    
    %Lp is the quantity to minimize (sq norm of parameters + slacks)
    L_T(k)=sum(diag(Beta'*H*Beta))/2 + para.C1*sum(L1)/2 + para.C2*sum(L2);

    eta=1; %initial step length
    %Loop where we keep alphas and modify betas
    while(L_T(k)>L_T(k-1))
        
        eta=eta/10; %modify step length
        i1=i1_a; %restore i1
        
        %the new betas are a combination of the current (sal1) and of the
        %previous iteration (Beta_a)
        Beta=zeros(size(Beta));
        Beta(i1,:)=eta*sal1+(1-eta)*Beta_a(i1,:);
        b=eta*b_sal1+(1-eta)*b_a;
        
        %recoumpte
        P = H*Beta + repmat(b, N, 1);
        E = Y - P;
        u=sqrt(sum(E.^2,2));
        i1=find(u>=para.epsi);
        
        L1 = zeros(size(u));
        L1(i1) = u(i1).^2-2*para.epsi*u(i1)+para.epsi^2;
        L2 = YS(i1,:) .* P(i1,:);
        i2 = L2 >= 0;
        L2(i2) = 0;
        L2 = -sum(L2,2);
        
        L_T(k)=sum(diag(Beta'*H*Beta))/2 + para.C1*sum(L1)/2 + para.C2*sum(L2);
        
        %stopping criterion #1
        if(eta<10^-16)
            L_T(k)=L_T(k-1)-10^-15;
            %save parameters
            Beta=Beta_a;
            b=b_a;
            i1 = i1_a;
            hacer=0; %stop loop
        end
    end
    
    a_a=a;
    a = 2*(u-para.epsi) ./ u;
    RMSE(k) = sqrt(mean(u.^2));

    %stopping criterion #2
    if((L_T(k-1)-L_T(k))/L_T(k-1) < para.tol)
        %fprintf('---> Iter:%4d, L_T:%15.7f, RMSE:%15.7f\n', k, L_T(k), RMSE(k));
        hacer = 0; %stop loop
    end

    %stopping criterion #3 - algorithm does not converge. (val = -1)   
    if(isempty(i1))
        %fprintf('---> Stop: algorithm does not converge (find no SVs).\n');    
        Beta = zeros(size(Beta));
        b = zeros(size(b));
        i1 = [];
        val = -1;
        hacer=0; %stop loop
    end
        
end

% save model
model.Beta = Beta;
model.b = b;
model.svindex = i1;
model.ker = para.ker;
model.par = para.par;

end

