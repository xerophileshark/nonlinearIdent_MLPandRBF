close
clear
clc
%% INITIALIZATION
% GLOBAL VARIABLES
t = 0:200; %descrete time vector (201 points)
max_epochs = 150;
lr = 0.015; %learning rate (mu)
Tr_percent = 0.8; %Percent of total training points
N = floor(Tr_percent*size(t, 2)); %number of training data
Ntst = size(t, 2) - N;%number of testing data
num_of_inputs = 3;
num_of_1stlayer_cells = 3;
num_of_2ndlayer_cells = 3;
num_of_outputs = 1;
% DEFINING THE SYSTEM DYNAMICS
alpha = 0.75;
beta = 1.5;
u = 0.3*sin(pi*t/12) + 0.5*cos(pi*t/5) + 0.1*sin(pi*t/49); %input signal to the dynamic system
y = zeros(1, size(t, 2));
for i = 3:size(t, 2) % y(0) and y(1) are zero.
    y(i) = alpha*( (y(i-1)*y(i-2)*(y(i-2)+beta)) / (1+(y(i-1))^2*(y(i-2))^2) + u(i-1));
end
    figure(1)
    subplot(1,2,1)
    plot(t, u, 'k-', 'LineWidth', 1.5)
    title('Input signal');
    xlabel('$k$','Interpreter','latex'); ylabel('$u(k)$','Interpreter','latex'); grid;
    subplot(1,2,2)
    plot(t, y, 'k-', 'LineWidth', 1.5)
    title('Output signal');
    xlabel('$k$','Interpreter','latex'); ylabel('$y(k)$','Interpreter','latex'); grid;
% PREPARING THE DATA-SET
% 1. MAKING INPUT-OUTPUT PAIRS BASED ON SYSTEM:
Bias1 = ones(num_of_1stlayer_cells, 201); %Bias for hidden layer 1
Bias2 = ones(num_of_2ndlayer_cells, 201); %Bias for hidden layer 2
X(1, :) = [0 y(1:200)]; %first input data (y(t-1))
X(2, :) = [0 0 y(1:199)]; %second input data (y(t-2))
X(3, :) = [0 u(1:200)]; %third input data (u(t-1))
% 2. SCALING INPUT-OUTPUT DATA:
X(1, :) = ((0.8 - (-0.8))/(max(X(1, :)) - min(X(1, :)))) * X(1, :) + ((max(X(1, :))*(-0.8)-min(X(1, :))*0.8)/(max(X(1, :))-min(X(1, :)))); %scale_factor*X(1, :) - offset_value
X(2, :) = ((0.8 - (-0.8))/(max(X(2, :)) - min(X(2, :)))) * X(2, :) + ((max(X(2, :))*(-0.8)-min(X(2, :))*0.8)/(max(X(2, :))-min(X(2, :)))); %scale_factor*X(2, :) - offset_value
X(3, :) = ((0.8 - (-0.8))/(max(X(3, :)) - min(X(3, :)))) * X(3, :) + ((max(X(3, :))*(-0.8)-min(X(3, :))*0.8)/(max(X(3, :))-min(X(3, :)))); %scale_factor*X(2, :) - offset_value
y = ((0.8 - (-0.8))/(max(y) - min(y))) * y + ((max(y)*(-0.8)-min(y)*0.8)/(max(y)-min(y))); %scale_factor*y - offset_value
    figure(2)
    subplot(2,2,1)
    plot(t, X(1, :), 'k-', 'LineWidth', 1.5)
    title('Scaled x_1 Input signal');
    xlabel('$n$','Interpreter','latex'); ylabel('$\bar{x}_1(n)$','Interpreter','latex'); grid;
    subplot(2,2,2)
    plot(t, X(2, :), 'k-', 'LineWidth', 1.5)
    title('Scaled x_2 Input signal');
    xlabel('$n$','Interpreter','latex'); ylabel('$\bar{x}_2(n)$','Interpreter','latex'); grid;
    subplot(2,2,3)
    plot(t, X(3, :), 'k-', 'LineWidth', 1.5)
    title('Scaled x_3 Input signal');
    xlabel('$n$','Interpreter','latex'); ylabel('$\bar{x}_3(n)$','Interpreter','latex'); grid;
    subplot(2,2,4)
    plot(t, y, 'k-', 'LineWidth', 1.5)
    title('Scaled Output signal');
    xlabel('$n$','Interpreter','latex'); ylabel('$\bar{y}(n)$','Interpreter','latex'); grid;
% DEFININIG DESIRED OUTPUTS
d = y; %desired value vector
% DEFINE AND RANDOM INITIALIZE WEIGHT VECTORS
W1Bias = [0; 0; 0];
W2Bias = [0; 0; 0];
Win_h1 = unifrnd(-1.3, 1.3, [num_of_inputs, num_of_1stlayer_cells])'; %Input to Hidden-Layer1 weights (W1)
Wh1_h2 = unifrnd(-1.3, 1.3, [num_of_1stlayer_cells, num_of_2ndlayer_cells])'; %Hidden-Layer1 to Hidden-Layer2 weights (W2)
Wh2_o = unifrnd(-1.3, 1.3, [num_of_2ndlayer_cells, num_of_outputs])'; %Hidden-Layer2 to output weights (W3)
O1 = [0;0;0]; %Output of first hidden layer
O2 = [0;0;0]; %Output of second hidden layer
Y = 0; %Output of neural network
%% TRAINING ALGORITHM
% <loop>
tic
epochs = max_epochs;
E = zeros(1, N);
Eave = zeros(1, epochs);
for i = 1:epochs
    % Random Permutation in training data:
    rndIndex = randperm(N);
    for n = 1:N
        % FORWARD PATH
        O1 = activation_func(Win_h1*X(:, rndIndex(n)) + W1Bias.*Bias1(:, rndIndex(n)));
        O2 = activation_func(Wh1_h2*O1 + W1Bias.*Bias2(:, rndIndex(n)));
        Y = Wh2_o*O2;%activation_func(Wh2_o*O2); %%%JUST LINEAR ACTIVATION FUNC
        % BACK PROPAGATION
        %1. Error calculations:
        e = d(rndIndex(n)) - Y; %e(n) vector containing e_k(n) k=1,2,...,m errors (m is number of outputs => here m=1)
        E(n) = sum(e'*e)/2;
        Eave(i) = Eave(i) + E(n);
        delta_3 = e .* 1;%(2*(1-Y.^2)); %%%JUST LINEAR ACTIVATION FUNC
        delta_2 = delta_3 .* (2*(1-O2.^2));
        delta_1 = delta_2 .* (2*(1-O1.^2));
        %2. Weight correction:
        DWh2_o = lr * delta_3 * O2'; %DW3
        DWh1_h2 = lr * delta_2 * O1'; %DW2
        DWin_h1 = lr * delta_1 * X(:, rndIndex(n))'; %DW1
        Wh2_o = Wh2_o + DWh2_o;
        Wh1_h2 = Wh1_h2 + DWh1_h2;
        Win_h1 = Win_h1 + DWin_h1;
    end
    Eave(i) = Eave(i) / N;
end
toc
% </loop>

%% TEST DATA
y_new = zeros(1, Ntst);
for i = N+1:N+Ntst
   o1 = activation_func(Win_h1*X(:, i) + W1Bias.*Bias1(:, i));
   o2 = activation_func(Wh1_h2*o1 + W1Bias.*Bias2(:, i));
   y_new(i-N) = activation_func(Wh2_o*o2);
end
    figure(3)
    plot(t(N+1:end), y_new(:), 'k--', 'LineWidth', 2)
    hold on
    plot(t(N+1:end), y(N+1:end), 'k-', 'LineWidth', 2)
    xlabel('$n$','Interpreter','latex'); ylabel('Normalized Amplitude','Interpreter','latex');
    legend('$\hat{y}(n)$', '$y(n)$', 'Interpreter', 'latex')

%% DISPLAY MSE
figure(4)
plot(1:epochs, Eave, 'k-', 'LineWidth', 2)
xlabel('\textbf{\textit{epoch}}','Interpreter','latex'); ylabel('\textbf{MSE}','Interpreter','latex');

%% FUNCTIONS
function y = activation_func(x)
    a = 2;
    y = (1 - exp(-a*x))./(1 + exp(-a*x)); %tanh(x);
%     y = x>0 .* x; %ReLU activation function
end