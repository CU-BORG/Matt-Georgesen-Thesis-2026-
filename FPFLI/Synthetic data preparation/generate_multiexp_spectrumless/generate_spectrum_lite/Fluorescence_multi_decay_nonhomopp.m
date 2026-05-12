function y = Fluorescence_multi_decay_nonhomopp(tau_Num,tau,f,N,h,IRF)
% tau_Num: number of lifetime components
% tau: lifetime values [tau1, tau2]
% f: fraction ratios [f1, f2] (must sum to 1)
% N: photon counts
% h: bin width (ns)
% IRF: instrument response function
% @author: mg

T = 256; % time bin number

% Generate time array and decay
% Decay starts at t=0 (bin 1), IRF convolution will shift it naturally
t = 1:T;
yo = zeros(size(t));

% Generate exponential decay starting at bin 1
% The IRF convolution will shift this to the correct position
for i = 1:tau_Num
    yo = yo + f(i)*exp(-t/(tau(i)/h));
end

I = IRF/max(IRF);
C=conv(I,yo);
lambda =C(1:length(t));
lambda_bound = max(lambda)*1.1;

% Acceptance-Rejection method to obtain desired distribution
count=zeros(N,1);
i=1;
while i<=N
    u1 = randi(256);
    u2 = rand(1);
    if u2<=lambda(u1)/lambda_bound
        count(i)=u1;
        i=i+1;
    end
end

Edges = 1:1:T+1;
[y,~]=histcounts(count,Edges);
%  %----adding background noise----%
% noise_ratio = 0.05
% for ii = 1: round(noise_ratio*N)
%     bin_position = randi(T);
%     y(bin_position) = y(bin_position)+1;
% end
end