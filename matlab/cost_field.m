clear all;
close all;

global ri rj a xB

xB = [5 5]';
% xi = [1 1]';
xj = [0.5 0.5 3; 
      3 1 1];
ri = 0.1;
rj = 0.1;
a = 5;

[X Y] = meshgrid(linspace(0,5,100));
cost_fun = 0*X;
gradU_fun = cost_fun;
gradV_fun = cost_fun;

for i=1:size(X,2)
    for j=1:size(X,1)
        xi = [X(i,j) Y(i,j)]';
        cost_fun(i,j) = cost(xi,xj);
        g = -grad(xi,xj);
        gradU_fun(i,j) = g(1);
        gradV_fun(i,j) = g(2);
    end
end


%% plots
figure;
surf(X,Y,cost_fun,'EdgeAlpha',0)
hold on
plot(xB(1),xB(2),'or','LineWidth',2)
for j=1:size(xj,2)
    plot(xj(1,j),xj(2,j),'ob','LineWidth',2)
end
% plot(xi(1),xi(2),'ok','LineWidth',2)
% xlim([0 6]); ylim([0 6]); 
% zlim([-inf,cost_fun(1,1)*1.1])
grid on;

figure;
% quiver(X,Y,gradU_fun,gradV_fun)
streamslice(X,Y,gradU_fun,gradV_fun)

%% Fnctions

function g = grad(xi,Xj)
    global ri rj a xB

    term1 = 2*(xi-xB);
    term2 = Xj*0;
    
    for j=1:size(Xj,2)
        xj = Xj(:,j);
        
        term2(:,j) = 1/(norm(xi-xj)-ri-rj) * (xi-xj)/norm(xi-xj);
    end
    term2 = -a*sum(term2,2);
    g = term1+term2;
    g = term2;

%     g = g/norm(g);
end

function c = cost(xi,Xj)
    global ri rj a xB
    dij = vecnorm(xi-Xj)-ri-rj;
    dij(dij<0)=0;
    c = norm(xi-xB)^2 - a*sum(log(dij));
    
end
