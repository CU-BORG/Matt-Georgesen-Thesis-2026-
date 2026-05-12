%% Generate Multi-Exponential Image Training Data (VARIABLE SAMPLE SIZES)
% Creates 256x256x256 training data cubes from 512x512 intensity images
% with GAUSSIAN-distributed lifetime values.
% GENERATES 10 FOLDERS WITH DIFFERENT NUMBERS OF IMAGES (1 to 10,000 in log steps)
%
% Output: 10 directories with varying dataset sizes:
%   - sample_size_01: 1 image
%   - sample_size_02: ~3 images
%   - sample_size_03: ~10 images
%   ...
%   - sample_size_10: 10,000 images
%
% Each .mat file contains:
%   - Hist: [256, 256, 256] - Fluorescence decay histograms
%   - tau_gt_components: [256, 256, 2] - Gaussian-distributed lifetime maps
%   - f_gt_components: [256, 256, 2] - Fraction maps (varying)
%   - tau_gt_avg: [256, 256] - Average lifetime maps
%   - Int: [256, 256] - Intensity image
%   - max_components: scalar - Number of components (2)
%
% Author: MG
% Date: 2025-12-18

clear; clc;

%% CENTRAL TAU VALUES WITH GAUSSIAN VARIATION
TAU_1_CENTER = 0.4;   % Short lifetime center (ns)
TAU_2_CENTER = 4.5;   % Long lifetime center (ns)

% Use Level 5 parameters (medium spectral overlap) for all datasets
TAU_1_STD = 0.30;  % Medium variation
TAU_2_STD = 1.00;  % Medium variation

%% Define 10 sample sizes in logarithmic steps from 1 to 10,000
SAMPLE_SIZE_LEVELS = 10;

% Logarithmically spaced sample sizes: 10^(0 to 4) in 10 steps
% This gives: 1, 3, 10, 32, 100, 316, 1000, 3162, 10000
log_min = 0;        % 10^0 = 1
log_max = 4;        % 10^4 = 10,000
log_steps = linspace(log_min, log_max, SAMPLE_SIZE_LEVELS);
SAMPLE_SIZES = round(10.^log_steps);

% Ensure first is exactly 1 and last is exactly 10,000
SAMPLE_SIZES(1) = 1;
SAMPLE_SIZES(end) = 10000;

fprintf('Sample sizes to generate:\n');
for i = 1:SAMPLE_SIZE_LEVELS
    fprintf('  Level %2d: %6d images\n', i, SAMPLE_SIZES(i));
end

%% Configuration
config.tau_Num = 2;              % Always 2 components
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins

% Path to 512x512 intensity images (PNG format)
config.image_source_path = 'C:\Users\mcg11923\Thesis\train';
config.image_pattern = '*.png';

% Tau parameters for Gaussian distributions (Level 5)
config.tau_center = [TAU_1_CENTER, TAU_2_CENTER];
config.tau_std = [TAU_1_STD, TAU_2_STD];

% Photon count scaling based on intensity
config.photon_scale_min = 500;    % Minimum photons for darkest pixels (10x increase)
config.photon_scale_max = 5000;   % Maximum photons for brightest pixels (10x increase)
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons

fprintf('\n=================================================================\n');
fprintf('Multi-Exponential Image Training Data Generation (VARIABLE SAMPLE SIZES)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Tau1: Gaussian(mu=%.2f ns, sigma=%.2f ns)\n', TAU_1_CENTER, TAU_1_STD);
fprintf('  Tau2: Gaussian(mu=%.2f ns, sigma=%.2f ns)\n', TAU_2_CENTER, TAU_2_STD);
fprintf('  Sample size levels: %d (from 1 to 10,000)\n', SAMPLE_SIZE_LEVELS);
fprintf('  Total images: %d\n', sum(SAMPLE_SIZES));
fprintf('  Input image size: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output image size: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);
fprintf('  Bin width: %.4f ns\n', config.bin_width);
fprintf('  Photon range: [%d, %d]\n', config.photon_scale_min, config.photon_scale_max);
fprintf('  Image source: %s\n', config.image_source_path);
fprintf('-----------------------------------------------------------------\n');

%% Load or generate IRF
irf_file = 'irf_measure.mat';
if exist(irf_file, 'file')
    fprintf('Loading IRF from %s...\n', irf_file);
    irf_data = load(irf_file);
    IRF = irf_data.irf;
else
    fprintf('Generating Gaussian IRF...\n');
    IRF = IRF_gaussian(14, config.bin_width, 0.1673);
end

% Normalize IRF
IRF = IRF / max(IRF);
fprintf('IRF loaded/generated successfully\n');

%% Load ALL available intensity images
fprintf('\nLoading 512x512 intensity images...\n');
image_files = dir(fullfile(config.image_source_path, config.image_pattern));

if isempty(image_files)
    error('No images found in %s with pattern %s', ...
        config.image_source_path, config.image_pattern);
end

n_available = length(image_files);
fprintf('Found %d available images\n', n_available);

% Check if we have enough images
max_needed = max(SAMPLE_SIZES);
if n_available < max_needed
    error('Need %d images but only %d available! Please provide more images.', ...
        max_needed, n_available);
end

%% Main loop: Generate data for each sample size level
fprintf('\n=================================================================\n');
fprintf('Starting generation for %d sample size levels...\n', SAMPLE_SIZE_LEVELS);
fprintf('=================================================================\n\n');

total_tic = tic;

for size_level = 1:SAMPLE_SIZE_LEVELS
    %% Set current sample size
    N_samples = SAMPLE_SIZES(size_level);

    %% Create size-level-specific output directory
    config.output_dir = sprintf('sample_size_%02d_n%06d', size_level, N_samples);

    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end

    fprintf('-----------------------------------------------------------------\n');
    fprintf('LEVEL %d/%d: Generating %d images\n', ...
        size_level, SAMPLE_SIZE_LEVELS, N_samples);
    fprintf('Output: %s\n', config.output_dir);
    fprintf('Progress: ');

    level_tic = tic;

    % Select random subset of images for this size
    % Use different random seed for each level to get different samples
    rng(size_level * 42);  % Reproducible but different per level
    selected_indices = randperm(n_available, N_samples);
    selected_images = image_files(selected_indices);

    for img_idx = 1:N_samples
        img_tic = tic;

        %% Load and process intensity image
        img_path = fullfile(selected_images(img_idx).folder, selected_images(img_idx).name);
        img_raw = imread(img_path);

        % Convert to grayscale if needed
        if size(img_raw, 3) == 3
            img_gray = rgb2gray(img_raw);
        else
            img_gray = img_raw;
        end

        % Ensure 512x512
        if size(img_gray, 1) ~= config.image_size_input || size(img_gray, 2) ~= config.image_size_input
            img_gray = imresize(img_gray, [config.image_size_input, config.image_size_input]);
        end

        % Store original and normalize to [0, 1]
        Int_512 = double(img_gray);
        img_normalized = double(img_gray) / 255.0;

        % Downsample to 256x256 using bilinear interpolation
        img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
        Int = img_256;

        %% Generate GAUSSIAN-distributed lifetime and VARYING fraction maps
        % Each pixel gets independently sampled tau and fraction values
        tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);
        f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

        % Generate per-pixel variations
        for y = 1:config.image_size_output
            for x = 1:config.image_size_output
                % Generate tau values from Gaussian distributions for this pixel
                for comp = 1:config.tau_Num
                    % Sample from Gaussian distribution
                    tau_sample = config.tau_center(comp) + config.tau_std(comp) * randn();

                    % Clamp to reasonable bounds
                    tau_min = max(0.1, config.tau_center(comp) - 3*config.tau_std(comp));
                    tau_max = config.tau_center(comp) + 3*config.tau_std(comp);
                    tau_map(y, x, comp) = max(tau_min, min(tau_max, tau_sample));
                end

                % Random fractions for this pixel (Dirichlet distribution)
                exp_rvs = -log(rand(1, config.tau_Num));
                f_pixel = exp_rvs / sum(exp_rvs);

                for comp = 1:config.tau_Num
                    f_map(y, x, comp) = f_pixel(comp);
                end
            end
        end

        % Store ground truth maps
        tau_gt_components = tau_map;  % [256, 256, 2]
        f_gt_components = f_map;      % [256, 256, 2]

        % Calculate average lifetime map
        tau_gt_avg = zeros(config.image_size_output, config.image_size_output);
        for comp = 1:config.tau_Num
            tau_gt_avg = tau_gt_avg + tau_map(:, :, comp) .* f_map(:, :, comp);
        end

        %% Generate fluorescence decay for each pixel
        hist_cube = zeros(config.image_size_output, config.image_size_output, config.bin_Num);

        for y = 1:config.image_size_output
            for x = 1:config.image_size_output
                % Get intensity at this pixel
                intensity = img_256(y, x);

                % Skip if below threshold
                if intensity < config.intensity_threshold
                    continue;
                end

                % Calculate photon count based on intensity
                N_photons = round(config.photon_scale_min + ...
                    intensity * (config.photon_scale_max - config.photon_scale_min));

                % Get lifetime components and fractions for this pixel
                tau_pixel = squeeze(tau_map(y, x, :))';
                f_pixel = squeeze(f_map(y, x, :))';

                % Generate multi-exponential decay
                decay = Fluorescence_multi_decay_nonhomopp(...
                    config.tau_Num, tau_pixel, f_pixel, N_photons, config.bin_width, IRF);

                hist_cube(y, x, :) = decay;
            end
        end

        % Store histogram cube
        Hist = hist_cube;  % [256, 256, 256]

        % Store max_components for compatibility
        max_components = config.tau_Num;

        %% Save individual file
        output_filename = sprintf('Sample_%06d_spectrumspectrum.mat', img_idx);
        output_path = fullfile(config.output_dir, output_filename);

        save(output_path, 'Hist', 'tau_gt_components', 'f_gt_components', ...
             'tau_gt_avg', 'Int', 'max_components', '-v7.3');

        img_time = toc(img_tic);

        % Progress update (more frequent for small datasets, less for large)
        if N_samples <= 10
            fprintf('%d (%.1fs) ', img_idx, img_time);
        elseif N_samples <= 100
            if mod(img_idx, 10) == 0
                fprintf('%d ', img_idx);
            end
        elseif N_samples <= 1000
            if mod(img_idx, 100) == 0
                fprintf('%d ', img_idx);
            end
        else  % > 1000
            if mod(img_idx, 1000) == 0
                fprintf('%d ', img_idx);
            end
        end

        if N_samples > 10 && mod(img_idx, 100) == 0
            fprintf('\n          ');
        end
    end

    level_time = toc(level_tic);
    fprintf('\n  Level %d completed in %.1f seconds (%.2f images/min)\n', ...
        size_level, level_time, N_samples / level_time * 60);
    fprintf('  Total images in this level: %d\n', N_samples);
    fprintf('  Approx size: %.1f MB\n\n', ...
        (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * N_samples);

end  % End of size_level loop

%% Final summary
total_time = toc(total_tic);
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (VARIABLE SAMPLE SIZES)!\n');
fprintf('=================================================================\n');
fprintf('Generated %d sample size levels:\n', SAMPLE_SIZE_LEVELS);
fprintf('  - Tau1: Gaussian(mu=%.2f ns, sigma=%.2f ns)\n', TAU_1_CENTER, TAU_1_STD);
fprintf('  - Tau2: Gaussian(mu=%.2f ns, sigma=%.2f ns)\n', TAU_2_CENTER, TAU_2_STD);
fprintf('  - Total images: %d\n', sum(SAMPLE_SIZES));
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes per file\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('\nOutput directories created:\n');
for i = 1:SAMPLE_SIZE_LEVELS
    dir_name = sprintf('  sample_size_%02d_n%06d/', i, SAMPLE_SIZES(i));
    fprintf('%s (%d images)\n', dir_name, SAMPLE_SIZES(i));
end
fprintf('\nFiles per directory: Sample_000001_spectrumspectrum.mat through Sample_######_spectrumspectrum.mat\n');
fprintf('Approx file size per image: %.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5);
fprintf('Total size: ~%.1f MB (~%.1f GB)\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * sum(SAMPLE_SIZES), ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * sum(SAMPLE_SIZES) / 1024);
fprintf('Total runtime: %.1f seconds (%.2f minutes, %.2f hours)\n', ...
    total_time, total_time/60, total_time/3600);
fprintf('\n=================================================================\n');
fprintf('SAMPLE SIZE PROGRESSION SUMMARY:\n');
fprintf('=================================================================\n');
fprintf('Level | Samples | Approx Size (MB) | Directory Name\n');
fprintf('-----------------------------------------------------------------\n');
for i = 1:SAMPLE_SIZE_LEVELS
    dir_name = sprintf('sample_size_%02d_n%06d', i, SAMPLE_SIZES(i));
    size_mb = (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * SAMPLE_SIZES(i);
    fprintf('%5d | %7d | %15.1f | %s\n', i, SAMPLE_SIZES(i), size_mb, dir_name);
end
