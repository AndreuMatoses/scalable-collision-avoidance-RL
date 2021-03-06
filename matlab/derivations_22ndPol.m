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

R = sym('R%d%d', [dim z_dim],'real');
% R1div =  Rderiv(theta(1)); R2div = Rderiv(theta(2)); R3div = Rderiv(theta(3));
% Rdiv = [R1div R2div R3div];
Sigma = sym('Sigma%d%d', [2 2],'real');
Sigma(2,1) = Sigma(1,2); 
% Sigma = inv(Sigma);
SigmaValues = [0.1, 0.01;0.01,0.15];