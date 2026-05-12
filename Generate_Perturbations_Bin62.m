%% Generate Perturbation Test Data for Bin 62 Model (12.5 ns, IRF @ bin 62)
% Creates synthetic FLIM data with various perturbations to test model robustness
% Matches temporal characteristics of the bin 62 trained model
%
% Groups generated:
%   1. baseline_normal - Standard bi-exponential
%   2. perturb_delayed_decay - Decay starts 1 ns later
%   3. perturb_tri_exponential - Three exponential components
%   4. perturb_flat_background - Constant background noise
%   5. perturb_50pct_photons - 50% photon count
%   6. perturb_10pct_photons - 10% photon count
%   7. perturb_1pct_photons - 1% photon count
%
% Author: MG
% Date: 2026-03-12

clear; clc; close all;

%% Add paths to required functions
addpath('C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite');
addpath(pwd);  % Add current directory for Fluorescence function

%% Configuration - Matching Bin 62 Model
config.bin_width = 0.048828;  % ns per bin (48.828 ps) - MATCHES REAL DATA
config.bin_Num = 256;         % Number of time bins
config.total_time = 12.5;     % ns total acquisition time
config.image_size = 256;      % Image size (256x256 pixels)
config.N_samples = 10;        % Number of samples per perturbation group

% IRF parameters - Matching bin 62 model
IRF_CENTER_BIN = 62;          % IRF peaks at bin 62 (3.027 ns)
IRF_FWHM = 0.120;             % 120 ps IRF width

% Lifetime parameters (matching training data distribution)
config.tau1_min = 0.1;        % Fast lifetime min (ns)
config.tau1_max = 1.0;        % Fast lifetime max (ns)
config.tau2_min = 1.5;        % Slow lifetime min (ns)
config.tau2_max = 4.5;        % Slow lifetime max (ns)
config.f_min = 0.1;           % Slow fraction min
config.f_max = 0.9;           % Slow fraction max

% Photon count parameters - Flat random distribution
config.photon_min = 50;       % Minimum photons per pixel
config.photon_max = 1000;     % Maximum photons per pixel

% Output base directory
config.output_base = 'E:\perturbations_bin62_12.5ns';

% Perturbation groups
GROUPS = {
    'baseline_normal',
    'perturb_delayed_decay',
    'perturb_tri_exponential',
    'perturb_flat_background',
    'perturb_50pct_photons',
    'perturb_10pct_photons',
    'perturb_1pct_photons'
};

fprintf('=================================================================\n');
fprintf('PERTURBATION DATA GENERATION FOR BIN 62 MODEL\n');
fprintf('=================================================================\n');
fprintf('Temporal Configuration:\n');
fprintf('  Bin width: %.6f ns (%.3f ps)\n', config.bin_width, config.bin_width * 1000);
fprintf('  Total time: %.1f ns\n', config.total_time);
fprintf('  Number of bins: %d\n', config.bin_Num);
fprintf('  IRF center: Bin %d (%.3f ns)\n', IRF_CENTER_BIN, IRF_CENTER_BIN * config.bin_width);
fprintf('  IRF FWHM: %.3f ns (%.0f ps)\n', IRF_FWHM, IRF_FWHM * 1000);
fprintf('\nLifetime Ranges:\n');
fprintf('  tau1: [%.1f, %.1f] ns\n', config.tau1_min, config.tau1_max);
fprintf('  tau2: [%.1f, %.1f] ns\n', config.tau2_min, config.tau2_max);
fprintf('  f (slow fraction): [%.1f, %.1f]\n', config.f_min, config.f_max);
fprintf('\nPhoton Range:\n');
fprintf('  Photons per pixel: [%d, %d] (flat random)\n', config.photon_min, config.photon_max);
fprintf('\nOutput:\n');
fprintf('  Base directory: %s\n', config.output_base);
fprintf('  Samples per group: %d\n', config.N_samples);
fprintf('  Total groups: %d\n', length(GROUPS));
fprintf('=================================================================\n\n');

%% Generate IRF
fprintf('Generating IRF...\n');
IRF = IRF_gaussian(IRF_CENTER_BIN, config.bin_width, IRF_FWHM);
IRF = IRF / max(IRF);  % Normalize
[~, irf_peak_bin] = max(IRF);
fprintf('  IRF peak at bin %d (%.3f ns)\n', irf_peak_bin, irf_peak_bin * config.bin_width);
fprintf('  IRF FWHM: %.3f ns\n\n', IRF_FWHM);

%% Main generation loop
total_tic = tic;

for group_idx = 1:length(GROUPS)
    group_name = GROUPS{group_idx};

    fprintf('=================================================================\n');
    fprintf('GROUP %d/%d: %s\n', group_idx, length(GROUPS), group_name);
    fprintf('=================================================================\n');

    % Create output directory
    output_dir = fullfile(config.output_base, group_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    group_tic = tic;

    % Special handling for baseline_normal - copy from training set
    if strcmp(group_name, 'baseline_normal')
        training_dir = 'E:\decay_peak_bin62_12.5ns_train';
        training_files = dir(fullfile(training_dir, 'Sample_*.mat'));

        if isempty(training_files)
            error('No training files found in %s', training_dir);
        end

        % Select 10 samples evenly distributed across training set
        n_training = length(training_files);
        if n_training < config.N_samples
            error('Not enough training samples (%d) to select %d baseline samples', n_training, config.N_samples);
        end

        % Evenly spaced indices (1, 11, 21, 31, 41, 51, 61, 71, 81, 91 for 100 samples)
        step = floor(n_training / config.N_samples);
        sample_indices = 1:step:(config.N_samples * step);
        sample_indices = sample_indices(1:config.N_samples);  % Ensure exactly N_samples

        fprintf('  Copying %d samples from training set (evenly distributed)...\n', config.N_samples);

        for idx = 1:config.N_samples
            copy_tic = tic;

            source_idx = sample_indices(idx);
            source_file = fullfile(training_dir, training_files(source_idx).name);
            dest_filename = sprintf('Sample_%03d_baseline_normal.mat', idx);
            dest_file = fullfile(output_dir, dest_filename);

            % Copy file
            copyfile(source_file, dest_file);

            copy_time = toc(copy_tic);
            fprintf('  [%d/%d] %s copied from training sample %d (%.2f s)\n', ...
                idx, config.N_samples, dest_filename, source_idx, copy_time);
        end

        group_time = toc(group_tic);
        fprintf('  Group completed in %.1f seconds (training data copied)\n\n', group_time);
        continue;  % Skip to next group
    end

    % For all other groups: generate samples
    for sample_idx = 1:config.N_samples
        sample_tic = tic;

        % Initialize output arrays
        Hist = zeros(config.image_size, config.image_size, config.bin_Num);
        tau_gt_components = zeros(config.image_size, config.image_size, 2);
        f_gt_components = zeros(config.image_size, config.image_size, 2);
        tau_gt_avg = zeros(config.image_size, config.image_size);
        Int = rand(config.image_size, config.image_size);  % Random intensity pattern

        % Generate decay for each pixel
        for y = 1:config.image_size
            for x = 1:config.image_size
                % Random lifetime parameters
                tau1 = config.tau1_min + (config.tau1_max - config.tau1_min) * rand();
                tau2 = config.tau2_min + (config.tau2_max - config.tau2_min) * rand();
                f_slow = config.f_min + (config.f_max - config.f_min) * rand();

                % Store ground truth
                tau_gt_components(y, x, 1) = tau1;
                tau_gt_components(y, x, 2) = tau2;
                f_gt_components(y, x, 1) = 1 - f_slow;  % Fast fraction
                f_gt_components(y, x, 2) = f_slow;      % Slow fraction
                tau_gt_avg(y, x) = (1 - f_slow) * tau1 + f_slow * tau2;

                % Base photon count (flat random)
                N_photons_base = randi([config.photon_min, config.photon_max]);

                % Apply group-specific modifications
                switch group_name
                    case 'perturb_delayed_decay'
                        % Delay decay by 1 ns (shift IRF right by ~20 bins)
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = N_photons_base;
                        delay_bins = round(1.0 / config.bin_width);  % 1 ns delay
                        IRF_used = circshift(IRF, delay_bins);

                    case 'perturb_tri_exponential'
                        % Add third component (tau3 = 2.5 ns, f3 = 20%)
                        tau3 = 2.5;
                        f3 = 0.2;
                        % Rescale other fractions
                        f1_scaled = (1 - f_slow) * (1 - f3);
                        f2_scaled = f_slow * (1 - f3);
                        tau = [tau1, tau2, tau3];
                        f = [f1_scaled, f2_scaled, f3];
                        N_photons = N_photons_base;
                        IRF_used = IRF;

                    case 'perturb_flat_background'
                        % Add 10% flat background
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = N_photons_base;
                        IRF_used = IRF;
                        background_fraction = 0.1;

                    case 'perturb_50pct_photons'
                        % 50% photon count
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_base * 0.5);
                        IRF_used = IRF;

                    case 'perturb_10pct_photons'
                        % 10% photon count
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_base * 0.1);
                        IRF_used = IRF;

                    case 'perturb_1pct_photons'
                        % 1% photon count
                        tau = [tau1, tau2];
                        f = [1 - f_slow, f_slow];
                        N_photons = round(N_photons_base * 0.01);
                        if N_photons < 1
                            N_photons = 1;
                        end
                        IRF_used = IRF;
                end

                % Generate decay
                decay = Fluorescence_multi_decay_nonhomopp(...
                    length(tau), tau, f, N_photons, config.bin_width, IRF_used);

                % Add flat background if needed
                if strcmp(group_name, 'perturb_flat_background')
                    background_counts = round(N_photons * background_fraction / config.bin_Num);
                    decay = decay + background_counts;
                end

                % Store
                Hist(y, x, :) = decay;
            end
        end

        % Save sample
        max_components = 2;  % Always report 2 components in ground truth
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

%% Final summary
total_time = toc(total_tic);
fprintf('=================================================================\n');
fprintf('PERTURBATION DATA GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Generated %d groups × %d samples = %d total .mat files\n', ...
    length(GROUPS), config.N_samples, length(GROUPS) * config.N_samples);
fprintf('Output directory: %s\n', config.output_base);
fprintf('Total runtime: %.1f seconds (%.2f minutes)\n', total_time, total_time/60);
fprintf('\nGroups created:\n');
for i = 1:length(GROUPS)
    fprintf('  %d. %s\n', i, GROUPS{i});
end
fprintf('\nTemporal characteristics (matches bin 62 model):\n');
fprintf('  Bin width: %.6f ns\n', config.bin_width);
fprintf('  IRF center: bin %d\n', IRF_CENTER_BIN);
fprintf('  IRF FWHM: %.3f ns\n', IRF_FWHM);
fprintf('  Total time: %.1f ns\n', config.total_time);
fprintf('=================================================================\n');
