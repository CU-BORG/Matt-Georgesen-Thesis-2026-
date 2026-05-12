%% Data Generation for Multi-Exponential NIII Training
% Generates synthetic FLIM images with multi-exponential decay components
% for training Neural Implicit Interpolation (NIII) models
%
% This script creates training data where each spatial location can have
% multiple lifetime components with associated fractions.
%
% Output files contain:
%   - Hist: Fluorescence decay histograms [H, W, 256]
%   - tau_gt_components: Lifetime component maps [H, W, n_components]
%   - f_gt_components: Fraction component maps [H, W, n_components]
%   - tau_gt_avg: Average lifetime map [H, W]
%   - component_masks: Active component indicators [H, W, n_components]
%
% Author:MG
% Date: 2025-09-30

clear; clc;

%% Configuration
config.N_samples = 1000;  % Number of training samples to generate
config.n_components = 2;  % Number of exponential components (2 or 3)

% HPA dataset path (modify to your path)
config.hpa_path = 'C:\Users\mcg11923\Thesis\train';

% Output path (modify to your path)
config.saved_path = 'C:\Users\mcg11923\Thesis\training_dataset_multiexp_s8';

% Lifetime ranges for components (in nanoseconds)
% These ranges ensure good separation between components
config.tau_ranges = [
    0.4, 2.0;   % Component 1: Short lifetime
    2.5, 4.5;   % Component 2: Long lifetime
    5.0, 7.0    % Component 3: Very long (only if n_components=3)
];

% Fraction distribution parameters
% alpha values for Dirichlet distribution (controls fraction bias)
% alpha=1 gives uniform distribution over simplex
config.fraction_alpha = ones(1, config.n_components);


fprintf('Multi-Exponential FLIM Training Data Generation\n');

fprintf('Configuration:\n');
fprintf('  Number of samples: %d\n', config.N_samples);
fprintf('  Number of components: %d\n', config.n_components);
fprintf('  HPA dataset path: %s\n', config.hpa_path);
fprintf('  Output path: %s\n', config.saved_path);
fprintf('\nLifetime ranges:\n');
for i = 1:config.n_components
    fprintf('  Component %d: [%.2f, %.2f] ns\n', ...
        i, config.tau_ranges(i,1), config.tau_ranges(i,2));
end
fprintf('=================================================================\n\n');

%% Setup
current_path = pwd;
cd(config.hpa_path);  % Navigate to HPA dataset

% Create output directory
if ~exist(config.saved_path, 'dir')
    mkdir(config.saved_path);
    fprintf('Created output directory: %s\n', config.saved_path);
end

%% Generate training samples
fprintf('Generating %d multi-exponential FLIM samples...\n\n', config.N_samples);

tic;
parfor i = 1:config.N_samples
    try
        %% Randomly select channel configuration
        % This determines which RGB channels to use for the image
        rand_val = rand();

        if rand_val < 0.2
            % Three channels
            RGB_channel = [1, 2, 4];
            n_channels = 3;
        elseif rand_val < 0.6
            % Two channels (version 1)
            RGB_channel = [1, 4];
            n_channels = 2;
        else
            % Two channels (version 2)
            RGB_channel = [2, 3];
            n_channels = 2;
        end

        %% Generate lifetime and fraction components for each channel
        tau_components = cell(1, n_channels);
        f_components = cell(1, n_channels);

        for ch = 1:n_channels
            % Generate random lifetime components
            tau = zeros(1, config.n_components);
            for comp = 1:config.n_components
                tau_min = config.tau_ranges(comp, 1);
                tau_max = config.tau_ranges(comp, 2);
                tau(comp) = tau_min + rand() * (tau_max - tau_min);
            end

            % Generate random fraction components
            % Simple method without Statistics Toolbox
            if config.n_components == 2
                % For 2 components: generate one random value, second is 1-first
                f1 = rand();
                f = [f1, 1-f1];
            else
                % For 3+ components: generate random values and normalize
                f = rand(1, config.n_components);
                f = f / sum(f);
            end

            % Store for this channel
            tau_components{ch} = tau;
            f_components{ch} = f;
        end

        %% Generate synthetic FLIM image with multi-exponential components
        GenSynFLI_MultiExp(i, tau_components, f_components, RGB_channel, config.saved_path);

        %% Progress indicator
        if mod(i, 100) == 0
            fprintf('Finished: %d/%d (%.1f%%)\n', i, config.N_samples, ...
                100*i/config.N_samples);
        end

    catch ME
        fprintf('Error processing sample %d: %s\n', i, ME.message);
    end
end

elapsed_time = toc;

%% Return to original path
cd(current_path);


%% Create sample visualization (first file only)
fprintf('\nCreating visualization of first sample...\n');

try
    % Load first generated file
    files = dir([config.saved_path, 'Sample_1_MultiExp*.mat']);
    if ~isempty(files)
        data = load([config.saved_path, files(1).name]);

        figure('Position', [100, 100, 1400, 800]);

        % Plot 1: Intensity image
        subplot(2, 4, 1);
        imagesc(data.Int);
        colorbar;
        title('Total Intensity');
        axis equal tight;

        % Plot 2: Average lifetime
        subplot(2, 4, 2);
        imagesc(data.tau_gt_avg);
        colorbar;
        title('Average Lifetime (ns)');
        axis equal tight;
        caxis([0, max(config.tau_ranges(:,2))]);

        % Plot 3-6: Individual component lifetimes
        for comp = 1:min(4, config.n_components)
            subplot(2, 4, 2 + comp);
            tau_comp = data.tau_gt_components(:, :, comp);
            mask = data.component_masks(:, :, comp);
            tau_comp(~mask) = NaN;
            imagesc(tau_comp);
            colorbar;
            title(sprintf('Component %d Lifetime (ns)', comp));
            axis equal tight;
            caxis([0, max(config.tau_ranges(:,2))]);
        end

        % Plot 7-8: Fraction components
        for comp = 1:min(2, config.n_components)
            subplot(2, 4, 6 + comp);
            f_comp = data.f_gt_components(:, :, comp);
            mask = data.component_masks(:, :, comp);
            f_comp(~mask) = NaN;
            imagesc(f_comp);
            colorbar;
            title(sprintf('Component %d Fraction', comp));
            axis equal tight;
            caxis([0, 1]);
        end

        sgtitle(sprintf('Multi-Exponential FLIM Sample (%d components)', ...
            config.n_components), 'FontSize', 14, 'FontWeight', 'bold');

        % Save figure
        fig_name = [config.saved_path, 'sample_visualization.png'];
        saveas(gcf, fig_name);
        fprintf('Visualization saved to: %s\n', fig_name);
    end
catch ME
    fprintf('Could not create visualization: %s\n', ME.message);
end

fprintf('\nAll done!\n');
