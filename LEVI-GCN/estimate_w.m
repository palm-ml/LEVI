function W = estimate_W( X )
fprintf(1,'Structural information discovery...\n');
N = size(X,1); %count of training samples
W = zeros(N, N);

% Finding the relationship via sparse reconstruction
for i=1:N
    A=X; 
    A(i,:)=[]; %extract the other examples, an overcomplete dictionary
    b=X(i,:)';
    lambda_max = norm( A*b, 'inf' );
    lambda = 0.01*lambda_max; 
    alpha = 0.5; %balance parameter
    rho = 1; %penalty parameter
    z = ADMM(A', b, lambda, rho, alpha);
    z(i+1:numel(z)+1)=z(i:end);
    z(i)=0;
    W(i,:) = z;            
end
end

function z = ADMM(A, b, lambda, rho, alpha)
global  ITER AB1 AB2
ITER = 1000;
AB1   = 1e-4;
AB2   = 1e-2;
[m, n] = size(A);
Atb = A'*b;
% Initialization
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

[L,U] = factor(A, rho);

for k = 1:ITER  % the scaled ADMM iterations
    q = Atb + rho*(z - u);   
    if( m >= n )    
       x = U \ (L \ q);
    else            
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);

    u = u + (x_hat - z);

    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*AB1 + AB2*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*AB1 + AB2*norm(rho*u);

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end



end

function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L,U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    
       L = chol( A'*A + rho*speye(n), 'lower' );
    else           
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end
    L = sparse(L);
    U = sparse(L');
end
