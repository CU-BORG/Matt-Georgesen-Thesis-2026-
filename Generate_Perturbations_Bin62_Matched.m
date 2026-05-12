%% Generate Parameter-Matched Perturbation Test Data for Bin 62 Model
% Creates perturbations using IDENTICAL ground truth parameters from baseline_normal
% Enables true paired comparison: same (tau1, tau2, f1, photons) with only perturbation applied
%
% Key difference from original script:
%   - baseline_normal: Copy from training data (already done)
%   - All perturbations: Load baseline, extract parameters, regenerate with perturbation
%
% Author: MG
% Date: 2026-03-12

clear; clc; close all;

%% Add paths to required functions
addpath('C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite');
addpath(pwd);

%% Configuration
config.bin_width = 0.048828;  % ns per bin
config.bin_Num = 256;
config.image_size = 256;
config.N_samples = 10;

% IRF parameters
IRF_CENTER_BIN = 62;
IRF_FWHM = 0.120;  % 120 ps

% Input/Output directories
config.baseline_dir = 'E:\perturbations_bin62_12.5ns\baseline_normal';
config.output_base = 'E:\perturbations_bin62_12.5ns_matched';

% Perturbation groups (excluding baseline_normal)
PERTURBATION_GROUPS = {
    'perturb_delayed_decay',
    'perturb_tri_exponential',
    'perturb_flat_background',
    'perturb_50pct_photons',
    'perturb_10pct_photons',
    'perturb_1pct_photons'
};

fprintf('=================================================================\n');
fprintf('PARAMETER-MATCHED PERTURBATION GENERATION\n');
fprintf('=================================================================\n');
fprintf('Strategy: Use SAME parameters as baseline_normal\n');
fprintf('  - Load baseline_normal samples (from training data)\n');
fprintf('  - Extract tau1, tau2, f1, photon counts pixel-by-pixel\n');
fprintf('  - Regenerate decays with IDENTICAL parameters + perturbation\n');
fprintf('  - Result: Pure perturbation effect, no parameter variation\n\n');
fprintf('Configuration:\n');
fprintf('  Baseline directory: %s\n', config.baseline_dir);
fprintf('  Output directory: %s\n', config.output_base);
fprintf('  Samples per group: %d\n', config.N_samples);
fprintf('  Perturbation groups: %d\n', length(PERTURBATION_GROUPS));
fprintf('=================================================================\n\n');

%% Verify baseline directory exists
if ~exist(config.baseline_dir, 'dir')
    error('Baseline directory not found: %s', config.baseline_dir);
end

baseline_files = dir(fullfile(config.baseline_dir, 'Sample_*.mat'));
if length(baseline_files) < config.N_samples
    error('Not enough baseline samples. Found %d, need %d', length(baseline_files), config.N_samples);
end

fprintf('Found %d baseline samples\n\n', length(baseline_files));

%% Generate IRF
fprintf('Generating IRF...\n');
IRF = IRF_gaussian(IRF_CENTER_BIN, config.bin_width, IRF_FWHM);
IRF = IRF / max(IRF);
fprintf('  IRF peak at bin %d (%.3f ns)\n', IRF_CENTER_BIN, IRF_CENTER_BIN * config.bin_width);
fprintf('  IRF FWHM: %.3f ns\n\n', IRF_FWHM);

%% Main generation loop
total_tic = tic;

for group_idx = 1:length(PERTURBATION_GROUPS)
    group_name = PERTURBATION_GROUPS{group_idx};

    fprintf('=================================================================\n');
    fprintf('GROUP %d/%d: %s\n', group_idx, length(PERTURBATION_GROUPS), group_name);
    fprintf('=================================================================\n');

    % Create output directory
    output_dir = fullfile(config.output_base, group_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    group_tic = tic;

    % Process each baseline sample
    for sample_idx = 1:config.N_samples
        sample_tic = tic;

        % Load corresponding baseline sample
        baseline_filename = sprintf('Sample_%03d_baseline_normal.mat', sample_idx);
        baseline_path = fullfile(config.baseline_dir, baseline_filename);

        if ~exist(baseline_path, 'file')
            error('Baseline file not found: %s', baseline_path);
        end

        baseline_data = load(baseline_path);

        % Extract ground truth parameters (pixel-wise)
        tau1_map = baseline_data.tau_gt_components(:, :, 1);
        tau2_map = baseline_data.tau_gt_components(:, :, 2);
        f1_map = baseline_data.f_gt_components(:, :, 1);
        f2_map = baseline_data.f_gt_components(:, :, 2);

        % Extract baseline photon counts (pixel-wise)
        baseline_photon_map = squeeze(sum(baseline_data.Hist, 3));

        % Initialize output arrays (copy structure from baseline)
        Hist_perturbed = zeros(config.image_size, config.image_size, config.bin_Num);
        tau_gt_components = baseline_data.tau_gt_components;  % SAME as baseline
        f_gt_components = baseline_data.f_gt_components;      % SAME as baseline
        tau_gt_avg = baseline_data.tau_gt_avg;                % SAME as baseline
        Int = baseline_data.Int;                              % SAME as baseline
        max_components = 2;

        % Regenerate decay for each pixel with perturbation
        for y = 1:config.image_size
            for x = 1:config.image_size
                % Extract baseline parameters for this pixel
                tau1 = tau1_map(y, x);
                tau2 = tau2_map(y, x);
                f_slow = f2_map(y, x);
                N_photons_baseline = baseline_photon_map(y, x);

                % Apply group-specific perturbation
                switch group_name
                    case 'perturb_delayed_decay'
                        % Same parameters, delayed IRF
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = N_photons_baseline;  % Same photon count
                        delay_bins = round(1.0 / config.bin_width);  % 1 ns delay
                        IRF_used = circshift(IRF, delay_bins);

                    case 'perturb_tri_exponential'
                        % Add 3rd component: tau3=2.5 ns with 20% weight
                        tau3 = 2.5;
                        f3 = 0.2;
                        % Rescale original fractions to sum to 0.8
                        f1_scaled = (1 - f_slow) * (1 - f3);
                        f2_scaled = f_slow * (1 - f3);
                        tau = [tau1, tau2, tau3];
                        f = [f1_scaled, f2_scaled, f3];
                        N_photons = N_photons_baseline;
                        IRF_used = IRF;

                    case 'perturb_flat_background'
                        % Same parameters, add 10% flat background
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = N_photons_baseline;
                        IRF_used = IRF;
                        background_fraction = 0.1;

                    case 'perturb_50pct_photons'
                        % Same parameters, 50% of baseline photons
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_baseline * 0.5);
                        IRF_used = IRF;

                    case 'perturb_10pct_photons'
                        % Same parameters, 10% of baseline photons
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_baseline * 0.1);
                        if N_photons < 1
                            N_photons = 1;
                        end
                        IRF_used = IRF;

                    case 'perturb_1pct_photons'
                        % Same parameters, 1% of baseline photons
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_baseline * 0.01);
                        if N_photons < 1
                            N_photons = 1;
                        end
                        IRF_used = IRF;
                end

                % Generate perturbed decay
                decay = Fluorescence_multi_decay_nonhomopp(...
                    length(tau), tau, f, N_photons, config.bin_width, IRF_used);

                % Add flat background if needed
                if strcmp(group_name, 'perturb_flat_background')
                    background_counts = round(N_photons * background_fraction / config.bin_Num);
                    decay = decay + background_counts;
                end

                % Store perturbed decay
                Hist_perturbed(y, x, :) = decay;
            end
        end

        % Save perturbed sample with SAME ground truth as baseline
        Hist = Hist_perturbed;  % Rename for consistency
        output_filename = sprintf('Sample_%03d_%s.mat', sample_idx, group_name);
        output_path = fullfile(output_dir, output_filename);

        save(output_path, 'Hist', 'tau_gt_components', 'f_gt_components', ...
             'tau_gt_avg', 'Int', 'max_components', '-v7.3');

        sample_time = toc(sample_tic);
        fprintf('  [%d/%d] %s saved (%.1f s)\n', sample_idx, config.N_samples, ...
            output_filename, sample_time);
    end

    group_time = toc(group_tic);
    fprintf('  Group completed in %.1f seconds (%.2f samples/min)\n\n', ...
        group_time, config.N_samples / group_time * 60);
end

%% Copy baseline_normal to matched directory for completeness
fprintf('=================================================================\n');
fprintf('Copying baseline_normal to matched directory...\n');
fprintf('=================================================================\n');

baseline_output_dir = fullfile(config.output_base, 'baseline_normal');
if ~exist(baseline_output_dir, 'dir')
    mkdir(baseline_output_dir);
end

for sample_idx = 1:config.N_samples
    source_file = fullfile(config.baseline_dir, sprintf('Sample_%03d_baseline_normal.mat', sample_idx));
    dest_file = fullfile(baseline_output_dir, sprintf('Sample_%03d_baseline_normal.mat', sample_idx));
    copyfile(source_file, dest_file);
    fprintf('  [%d/%d] Copied baseline sample %d\n', sample_idx, config.N_samples, sample_idx);
end

fprintf('  Baseline copy complete\n\n');

%% Final summary
total_time = toc(total_tic);
fprintf('=================================================================\n');
fprintf('PARAMETER-MATCHED PERTURBATION GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Generated %d perturbation groups + 1 baseline = %d total groups\n', ...
    length(PERTURBATION_GROUPS), length(PERTURBATION_GROUPS) + 1);
fprintf('Samples per group: %d\n', config.N_samples);
fprintf('Total files: %d\n', (length(PERTURBATION_GROUPS) + 1) * config.N_samples);
fprintf('Output directory: %s\n', config.output_base);
fprintf('Total runtime: %.1f seconds (%.2f minutes)\n', total_time, total_time/60);
fprintf('\nKey feature: All perturbations use IDENTICAL (tau1, tau2, f1) as baseline\n');
fprintf('  → Enables pixel-wise paired comparison\n');
fprintf('  → MAE differences purely reflect perturbation effect\n');
fprintf('  → Can compute error difference maps: MAE_perturb - MAE_baseline\n');
fprintf('=================================================================\n');
