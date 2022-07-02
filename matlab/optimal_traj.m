clear all;
close all;

T = 30;
i_agents = 3;
ri =[1 1 1]*0; % drone diameter
x_size = 2*i_agents; % dim* n_agents
X_size = T*x_size;
xF = [1 0 4 0 6 0]';
x0 = [-1 0 -2 0 -3 0]';

normalizing_r = ((xF(1:2:end)-xF(1:2:end)').^2+(xF(2:2:end)-xF(2:2:end)').^2).^0.5;
normalizing_r = mink(normalizing_r,2,2);
normalizing_r = normalizing_r(:,2);

XB=kron(ones(T,1),xF);
ineqA = eye(T) - (triu(ones(T),-1)-triu(ones(T)));
ineqA = kron(ineqA,eye(x_size));
ineqA = [ineqA;-ineqA];

u_max = 0.4;
ineqB = ones(X_size*2,1)*u_max;
ineqB(1:x_size) = u_max + x0;
ineqB(X_size+1:X_size+x_size) = u_max - x0;

X0 = rand(X_size,1);
% load sol
% X0 = sol;

fun = @(x) f(x,XB,x_size,T,normalizing_r,ri);

[sol,fval] = fmincon(fun,X0,ineqA,ineqB)

%% separate solution:
sol_re = reshape(sol,x_size,T);
x_sol = sol_re(1:2:end,:);
y_sol = sol_re(2:2:end,:);

for t=2:size(x_sol,2)
    figure(1)
    plot([x0(1:2:end)';x_sol(:,1:t-1)'],[x0(2:2:end)';y_sol(:,1:t-1)'],'-^')
    hold on
    plot(xF(1:2:end)',xF(2:2:end)','x')
    grid on;
    plot(x0(1:2:end),x0(2:2:end),'o')
    hold off;
    
    pause(0.2)
    drawnow
end



%% cost function
function obj = f(X,XF,x_size,T,normalazing_r,ri)

b = 0.4;

% goal cost
obj_end = (X-XF)'*(X-XF);

% collision cost
xt = reshape(X,x_size,T);
x_t = xt(1:2:end,:);
y_t = xt(2:2:end,:);

obj_colision = 0;
for t = 1:size(xt,2)
    x_diff = x_t(:,t)-x_t(:,t)';
    y_diff = y_t(:,t)-y_t(:,t)';
    
    D = (x_diff.^2 + y_diff.^2).^0.5 + (-ri-ri').*(1-eye(3));
    D_clip = min(1,D./normalazing_r)+eye(size(D,1));
    
    obj_colision = obj_colision -b*sum(sum(log(D_clip)));
    
end

obj=obj_end + obj_colision;
end