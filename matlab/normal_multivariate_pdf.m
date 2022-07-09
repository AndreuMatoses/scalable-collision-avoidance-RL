clear all;

mu = [0 0];
sigma = 0.1;
Sigma = [sigma 0; 0 sigma];


delta_x = 0.05;

x1 = -3:delta_x:3;
x2 = -3:delta_x:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y = mvnpdf(X,mu,Sigma);
y = reshape(y,length(x2),length(x1));
surf(x1,x2,y,'EdgeAlpha',0.1)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
% axis([-3 3 -3 3 0 0.4])
xlabel('x1')
ylabel('x2')
zlabel('Probability Density')

volumen = sum(sum(y*delta_x^2))
