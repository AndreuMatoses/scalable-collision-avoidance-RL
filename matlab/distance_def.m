%% Visualization of the distance between agents (variables)

clear all;

di_hat = 5;
max_dist = 7;
x_dist = linspace(0,max_dist,100);
% capping to d_hat, i,e dij = min(di_hat, ||xi-xj||-li-lj)
dij = x_dist;
dij(x_dist>di_hat)=di_hat;
dij_norm = di_hat./dij;
log_d = log(dij_norm);

%% plots
figure;
grid on;
subplot(3,1,1)
plot(x_dist,dij)
title(['$\hat{d}_i =' num2str(di_hat) '$'])
xlabel('$||x_i-x_j||-l_i-l_j$')
ylabel('$d_{ij} = min(\hat{d}_i, ||x_i-x_j||-l_i-l_j)$')
grid on;

subplot(3,1,2)
plot(x_dist,dij_norm)
xlabel('$||x_i-x_j||-l_i-l_j$')
ylabel('$\frac{\hat{d}_i}{d_{ij}}$')
grid on;

subplot(3,1,3)
plot(x_dist,log_d)
xlabel('$||x_i-x_j||-l_i-l_j$')
ylabel('$log(\frac{\hat{d}_i}{d_{ij}})$')
grid on;

