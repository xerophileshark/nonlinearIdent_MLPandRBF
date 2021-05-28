close
clear
clc
%% PREPARE INPUT DATA
t = 0:200; %descrete time vector (201 points)
alpha = 0.75;
beta = 1.5;
u = 0.3*sin(pi*t/12) + 0.5*cos(pi*t/5) + 0.1*sin(pi*t/49); %input signal to the dynamic system
y = zeros(1, size(t, 2));
Tr_percent = 0.8;
N = floor(Tr_percent*size(t, 2)); %number of training data == number of hidden cells
Ntst = size(t, 2) - N;%number of testing data
for i = 3:size(t, 2) % y(0) and y(1) are zero.
    y(i) = alpha*( (y(i-1)*y(i-2)*(y(i-2)+beta)) / (1+(y(i-1))^2*(y(i-2))^2) + u(i-1));
end
% PREPARING THE DATA-SET
% 1. MAKING INPUT-OUTPUT PAIRS BASED ON SYSTEM:
X(1, :) = [0 y(1:200)]; %first input data (y(t-1))
X(2, :) = [0 0 y(1:199)]; %second input data (y(t-2))
X(3, :) = [0 u(1:200)]; %third input data (u(t-1))
% 2. SCALING INPUT-OUTPUT DATA (NORMALIZATION):
X(1, :) = ((0.8 - (-0.8))/(max(X(1, :)) - min(X(1, :)))) * X(1, :) + ((max(X(1, :))*(-0.8)-min(X(1, :))*0.8)/(max(X(1, :))-min(X(1, :)))); %scale_factor*X(1, :) - offset_value
X(2, :) = ((0.8 - (-0.8))/(max(X(2, :)) - min(X(2, :)))) * X(2, :) + ((max(X(2, :))*(-0.8)-min(X(2, :))*0.8)/(max(X(2, :))-min(X(2, :)))); %scale_factor*X(2, :) - offset_value
X(3, :) = ((0.8 - (-0.8))/(max(X(3, :)) - min(X(3, :)))) * X(3, :) + ((max(X(3, :))*(-0.8)-min(X(3, :))*0.8)/(max(X(3, :))-min(X(3, :)))); %scale_factor*X(2, :) - offset_value
y = ((0.8 - (-0.8))/(max(y) - min(y))) * y + ((max(y)*(-0.8)-min(y)*0.8)/(max(y)-min(y))); %scale_factor*y - offset_value
d = y'; %desired value vector

clear i alpha beta t

%% 0. TIKHONOV METHOD: N CENTERS & N WEIGHTS
% CALCULATE WEIGHTS:
tic
W = zeros(N, 1);

G = zeros(N);
for i=1:N
    for j=1:N
        G(i, j) = Green(X(:, i), X(:, j), 1);
    end
end

lambda = 0.1;
W = inv(G + lambda*eye(N)) * d(1:N);
toc
% TEST DATA:
y_new1 = zeros(1, Ntst);
for i=N+1:N+Ntst
    for j=1:N
        y_new1(i - N) = y_new1(i - N) + W(j) * Green(X(:, i), X(:, j), 1);
    end
end

figure(1)
plot(d(N+1:N+Ntst),  'b-', 'LineWidth', 1.5);
hold on
plot(y_new1, 'r--', 'LineWidth', 1.5);
title('TIKHONOV METHOD: N CENTERS & N WEIGHTS')
xlabel('$n$','Interpreter','latex'); ylabel('$Amplityde$','Interpreter','latex');
legend('$y(t)$', '$\hat{y}(t)$', 'Interpreter', 'latex')
figure(2)
stem(W)
title('Final weights (TIKHONOV METHOD)')
xlabel('$i$','Interpreter','latex', 'FontSize', 13); ylabel('$w_i$','Interpreter','latex', 'FontSize', 14);
grid

clear i j lambda W G

%% 1. RBF RANDOM CENTERS & M WEIGHTS
M = 10; %Number of hidden cells
t = unifrnd(-1, 1, [3, M]); %Random centers vector
dmax = norm(max(t) - min(t));
sigma = dmax / sqrt(2*M);
green_coef = 1/2/(sigma^2);

% TRAIN WEIGHTS:
tic
W = zeros(M, 1);

G = zeros(N, M);
for i=1:N
    for j=1:M
        G(i, j) = Green(X(:, i), t(:, j), green_coef);
    end
end

G0 = zeros(M);
for i=1:M
    for j=1:M
        G0(i, j) = Green(t(:, i), t(:, j), green_coef);
    end
end

lambda = 0.25;
W = inv(G'*G + lambda*G0) * G'*d(1:N);
toc

% TEST DATA:
y_new2 = zeros(1, Ntst);
for i=N+1:N+Ntst
    for j=1:M
        y_new2(i - N) = y_new2(i - N) + W(j) * Green(X(:, i), t(:, j), green_coef);
    end
end

figure(3)
plot(d(N+1:N+Ntst),  'b-', 'LineWidth', 1.5);
hold on
plot(y_new2, 'r--', 'LineWidth', 1.5);
title('RANDOM CENTERS & M = 10 HIDDEN CELLS and WEIGHTS')
xlabel('$n$','Interpreter','latex'); ylabel('$Amplityde$','Interpreter','latex');
legend('$y(t)$', '$\hat{y}(t)$', 'Interpreter', 'latex')
disp('Final weights (TIKHONOV-M + RANDOM): ')
disp(W)
clear i j lambda W G M t dmax

%% 2. RBF K-MEANS CENTERS & M WEIGHTS RLS
M = 10; %Number of hidden cells & centers of clusters
t_k = unifrnd(-0.7, 0.7, [3, M]); %Initializing t_k(0)s (k = 1,2,...,M) vector randomly
figure(4)
subplot(1,2,1)
for i = 1:size(t_k, 2)
    plot3(t_k(1, i), t_k(2, i), t_k(3, i), 'mo', 'LineWidth', 2)
    hold on
end
plotZeroPlane()
hold off
view(70, 10)
grid
title('Centers befor K-means')
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');

% K-MEANS ALGORITHM FOR CENTERS:
disp('K-means algorithm time:')
tic
rndIndex = randperm(N); %random index for random sample picking
t_k_new = zeros(3, M);
while 1
    meanOfCenters = zeros(3, M);
    N_t_k_new = zeros(1, M);
    for n=1:N
        k = K(X(:, rndIndex(n)), t_k);
        meanOfCenters(:, k) = meanOfCenters(:, k) + X(:, rndIndex(n));
        N_t_k_new(k) = N_t_k_new(k) + 1;
        %calculate t_k_new immediately after each point is assigned to cluster k:
        t_k_new(:, k) = meanOfCenters(:, k)  / N_t_k_new(k);
    end
    %Update t_k:
    if t_k_new == t_k
        %end of clustering
        break
    else
        %update t_k
        t_k = t_k_new;
        %show update animation:
        figure(4)
        subplot(1,2,2);
        for i=1:N
            plot3(X(1, i), X(2, i), X(3, i), 'r+')
            hold on
        end
        plotZeroPlane()
        for i = 1:size(t_k, 2)
        %     plot3(t_k(1, i), t_k(2, i), t_k(3, i), 'mo', 'LineWidth', 2)
        plot3(t_k(1, i), t_k(2, i), t_k(3, i), 'mo', 'LineWidth', 2)
        end
        hold off
        title('Centers after K-means')
        view(70, 10)
        grid
        pause(0.005)
    end
end
% % % [~, C] = kmeans(X', M); %Matlab's builtin function for k-means algorithm
% % % C = C';
% PLOT MEANS:
toc
figure(4)
subplot(1,2,2);
for i=1:N
    plot3(X(1, i), X(2, i), X(3, i), 'r+')
    hold on
end
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
plotZeroPlane()
for i = 1:size(t_k, 2)
%     plot3(t_k(1, i), t_k(2, i), t_k(3, i), 'mo', 'LineWidth', 2)
plot3(t_k(1, i), t_k(2, i), t_k(3, i), 'mo', 'LineWidth', 2)
end
hold off
title('Centers after K-means')
view(70, 10)
grid

% RLS ALGORITHM FOR WEIGHTS TRAINING:
disp('K-means algorithm time:')
tic
lambda = 0.1;
P_n = lambda * eye(M);
g_n = zeros(M, 1);
alpha_n = 0; %
PHI = zeros(M, 1);
W = zeros(M, 1);
rndIndex = randperm(N); %random index for random sample picking
max_epochs = 150;
epochs = max_epochs;
E = zeros(1, N);
Eave = zeros(1, epochs);
for epoch=1:epochs
    for n=1:N
        for i=1:M
            PHI(i) = Green(X(:, rndIndex(n)), t_k(:, i), green_coef);
        end
        P_n = P_n - ((P_n*(PHI*PHI')*P_n)/(1+PHI'*P_n*PHI));
        g_n = P_n*PHI;
        alpha_n = d(rndIndex(n)) - W'*PHI;
        W = W + g_n*alpha_n;
        %Calculations for MSE:
        Y = W'*PHI;
        e = d(rndIndex(n)) - Y;
        E(n) = sum(e'*e)/2;
        Eave(epoch) = Eave(epoch) + E(n);
    end
    Eave(epoch) = Eave(epoch) / N;
end
toc
figure(5) %Plot MSE
plot(1:epochs, Eave, 'k-', 'LineWidth', 2)
xlabel('\textbf{\textit{epoch}}','Interpreter','latex'); ylabel('\textbf{MSE}','Interpreter','latex');
title('Mean Square Error for RLS training')

% TEST DATA:
y_new3 = zeros(1, Ntst);
for i=N+1:N+Ntst
    for j=1:M
        y_new3(i - N) = y_new3(i - N) + W(j) * Green(X(:, i), t_k(:, j), green_coef);
    end
end

figure(6)
plot(d(N+1:N+Ntst),  'b-', 'LineWidth', 1.5);
hold on
plot(y_new3, 'r--', 'LineWidth', 1.5);
title('K-MEANS CENTERS & LMS WEIGHTS')
xlabel('$t$','Interpreter','latex'); ylabel('$STD$','Interpreter','latex');
legend('$y(t)$', '$\hat{y}(t)$', 'Interpreter', 'latex')
disp('Final weights (K-MEANS + RLS): ')
disp(W)
clear n

%% 3. RBF GD LEARNING FOR Weights, Centers and Widths of Green functions


%%
function g = Green(x1, x2, sigma)
    g = exp(-sigma * norm(x1 - x2)^2);
end

function out = K(x, centers)
    arg = 1;
    min = norm(x - centers(:, 1));
    for k=2:size(centers, 2)
        Norm = norm(x - centers(:, k));
        if Norm <= min
            min = Norm;
            arg = k;
        end
    end
    out = arg;
end

function plotZeroPlane()
    p1 = [-2 -2 0];
    p2 = [2 -2 0];
    p3 = [2 2 0];
    p4 = [-2 2 0]; 
    x = [p1(1) p2(1) p3(1) p4(1)];
    y = [p1(2) p2(2) p3(2) p4(2)];
    z = [p1(3) p2(3) p3(3) p4(3)];
    fill3(x, y, z, 'c');
end