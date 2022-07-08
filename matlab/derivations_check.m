clear all;

rng(1)

z_dim = 3;
z = sym('z%d%d', [z_dim 1],'real');
zValues = rand(size(z))*5;
a = sym('a%d%d', [2 1],'real');
aValues = rand(size(a));
theta = sym('theta%d%d', [2 z_dim],'real');
thetaValues = rand(size(theta));
Sigma = sym('Sigma%d%d', [2 2],'real');
SigmaValues = rand(size(Sigma));
SigmaValues = SigmaValues.*SigmaValues';
val = @(eqq) double(subs(eqq,[z(:)',theta(:)',a(:)',Sigma(:)'],[zValues(:)',thetaValues(:)',aValues(:)',SigmaValues(:)']));


eqq1 = -1/2 * (a-theta*z)' * (Sigma) * (a-theta*z);
eqq2 =-1/2*z'*theta'*Sigma*theta*z + a'*Sigma*theta*z - 1/2*a'*Sigma*a;

for d = 1:size(a)
    for m = 1:z_dim
       grad1(d,m) = diff(eqq1,theta(d,m)); 
       grad2(d,m) = diff(eqq2,theta(d,m));
    end
end

[val(grad1), val(grad2)]

for m=1:z_dim
    IIgrad(:,m) = (a'*Sigma*z(m));
end


