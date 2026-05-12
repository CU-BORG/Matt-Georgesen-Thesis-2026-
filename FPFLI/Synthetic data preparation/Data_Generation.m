clear
clc

current_path = pwd;
cd C:\Users\mcg11923\Thesis\train % HPA datasets
saved_path = 'E:\trainingbigset\'; % saved path

if ~exist(saved_path, 'dir')
    mkdir(saved_path)
end

a = dir('*.png');

N = 15000;

tic
parfor i = 1:N
    if rand()<0.2
        RGB_channel = [1,2,4];
        tau = 0.4 + rand(1,3)*4.5;
    elseif rand()<0.6 && rand()>=0.2
        RGB_channel = [1,4];
        tau = 0.4 + rand(1,2)*4.5;
    else
        RGB_channel = [2,3];
        tau = 0.4 + rand(1,2)*4.5;
    end
    GenSynFLI(i,tau,RGB_channel,saved_path)
    disp(['Finished: ',num2str(i),'/',num2str(N)])
end
toc
cd (current_path)