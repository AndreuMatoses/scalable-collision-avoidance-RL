clear all;

rng(1)

z_dim = 6;
z = sym('z%d%d', [z_dim 1],'real');
% zValues = rand(size(z))*5;
zValues = [1,2,3,4,5,6]';
a = sym('a%d%d', [2 1],'real');
% aValues = rand(size(a));
aValues = [0.5,1]';
theta = sym('theta%d%d', [2 z_dim],'real');
% thetaValues = rand(size(theta));
% thetaValues(:,5:6)=0;
thetaValues = [1,0.5,1,0.5,1,0.5;2,0.6,2,0.6,2,0.6];
Sigma = sym('Sigma%d%d', [2 2],'real');
SigmaValues = [1,0.01;0.01,1.5];
val = @(eqq) double(subs(eqq,[z(:)',theta(:)',a(:)',Sigma(:)'],[zValues(:)',thetaValues(:)',aValues(:)',SigmaValues(:)']));

d = 2;
original = log(1/((2*pi)^(d/2)*sqrt(det(Sigma)))*exp(-1/2*(a-theta*z)'*inv(Sigma)*(a-theta*z)));

eqq1 = -1/2 * (a-theta*z)' * inv(Sigma) * (a-theta*z);
eqq2 =-1/2*z'*theta'*inv(Sigma)*theta*z + a'*inv(Sigma)*theta*z - 1/2*a'*inv(Sigma)*a;
I = -1/2*z'*theta'*inv(Sigma)*theta*z;
II = a'*inv(Sigma)*theta*z;

% for m=1:z_dim
%     dum = theta(:,m);
%     eval(['theta' num2str(m) ' = dum;']);
% end

gra = @(scalar) arrayfun(@(s,v) val(diff(s,v)),scalar*ones(size(theta)),theta);

%
grad1 = gra(eqq1);
grad2 = gra(eqq2);

IIgrad = val(inv(Sigma)*a*z');
Igrad = val(-inv(Sigma)*theta*z*z');
total_grad = val(inv(Sigma)*(a-theta*z)*z')
grad1






