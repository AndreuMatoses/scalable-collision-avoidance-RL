clear all;

rng(1)

z_dim = 6;
dim = 2;
z = sym('z%d%d', [z_dim 1],'real');
z1 = z(1:2); z2 = z(3:4); z3 = z(5:6);
zValues = rand(size(z))*5;
a = sym('a%d%d', [2 1],'real');
aValues = rand(size(a));
theta = sym('theta%d%d', [1 z_dim/dim],'real');
thetaValues = rand(size(theta));
R2d = @(angle) [cos(angle) -sin(angle); sin(angle) cos(angle)];
Rderiv = @(angle) [-sin(angle) -cos(angle); cos(angle) -sin(angle)];
R1 = R2d(theta(1)); R2 = R2d(theta(2)); R3 = R2d(theta(3)); 
R = [R1 R2 R3];
R1div =  Rderiv(theta(1)); R2div = Rderiv(theta(2)); R3div = Rderiv(theta(3));
Rdiv = [R1div R2div R3div];
Sigma = sym('Sigma%d%d', [2 2],'real');
Sigma(2,1) = Sigma(1,2); 
% Sigma = inv(Sigma);
SigmaValues = [0.1, 0.01;0.01,0.15];

val = @(eqq) double(subs(eqq,[z(:)',theta(:)',a(:)',Sigma(:)'],[zValues(:)',thetaValues(:)',aValues(:)',SigmaValues(:)']));

d = 2;
original = log(1/((2*pi)^(d/2)*sqrt(det(Sigma)))*exp(-1/2*(a-R*z)'*(Sigma)*(a-R*z)));

eqq1 = -1/2 * (a-R*z)' * (Sigma) * (a-R*z);
eqq2 =-1/2*z'*R'*(Sigma)*R*z + a'*(Sigma)*R*z - 1/2*a'*(Sigma)*a;
I = -1/2*z'*R'*(Sigma)*R*z;
II = a'*(Sigma)*R*z;

% for m=1:z_dim
%     dum = theta(:,m);
%     eval(['theta' num2str(m) ' = dum;']);
% end

Vgra = @(scalar) arrayfun(@(s,v) val(diff(s,v)),scalar*ones(size(theta)),theta);
gra = @(scalar) arrayfun(@(s,v) diff(s,v),scalar*ones(size(theta)),theta);


%
grad1 = gra(eqq1);
grad2 = gra(eqq2);

IIgrad = [val(a'*Sigma*R1div*z1), val(a'*Sigma*R2div*z2), val(a'*Sigma*R3div*z3); Vgra(II)]
% Igrad = val(-(Sigma)*theta*z*z');
% total_grad = val((Sigma)*(a-theta*z)*z')
grad1


%% loops
tot = 0;
l = 2;
zz = rand(l,1);
RR = rand(l,l);
SS = rand(l,l);

for i=1:l
    for j =1:l
        tot = tot + z1(i)*R1(i,j)*Sigma(i,j)*R1(j,i)*z1(j);
    end
end
tot




