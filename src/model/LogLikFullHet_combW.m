function L=LogLikFullHet_combW(b,W,Y,X)
% Panzica Roberto 
% concentrated loglikelihood fct
% b is the parameter vector to estimate (b=[delta,rho_vector]
% the delta has F dimension, rho_vector has dimension equal to N;
% W is a structure array containg F spatial matrices (F)x1; each W{f,1}is NXN where f=1:F.
% Y is TxN vector of returns
% X TXK explanatory variables.

% 
F=size(W,1);
d=sum(exp(b(2:F)));

% compute SAR innovations and covariance
Et=Y*0;
if F==1
A=eye(size(Y,2))-diag(b)*W;
elseif F>1
 %%%%A=eye(size(Y,2))-diag(b(F+1:end))*((1/(1+d))*W{1,1}+(exp(b(2))/(1+d))*W{2,1}+(exp(b(3))/(1+d))*W{3,1});;
 %%%% I dinamically write the equation (4) we impose that sum of delta
 %%%% parameters is equal to one.
 %%%% parametrized
 s1=[];
    for hh=2:F
      s= strcat('(exp(b(',num2str(hh),'))/(1+d))*W{',num2str(hh),',1}','+');
      s1=[s1,s];
    end
    s1=s1(1:end-1);
    ss=strcat('A=eye(size(Y,2))-diag(b(F+1:end))*((1/(1+d))*W{1,1}+',s1,');');
    eval(ss);
end
% demeaning of returns and explanatory variables. 
Y1=Y-repmat(mean(Y),size(Y,1),1);
X1=X-repmat(mean(X),size(X,1),1);
for j=1:size(Y1,1);
    Et(j,:)=(A*(Y1(j,:)'))';
end
XtX=(X1'*X1);
XtY=(X1'*Et);
B=(XtX\XtY)';   % Estimated intercept and betas 
Et2=Et-X1*B';

O=diag(diag(cov(Et2)));
l=0;
for j=1:size(Y1,1);
    l=l+log(det(A))-0.5*log(det(O))-0.5*(((Et2(j,:))/O)*Et2(j,:)');
end
L=-l;
end
