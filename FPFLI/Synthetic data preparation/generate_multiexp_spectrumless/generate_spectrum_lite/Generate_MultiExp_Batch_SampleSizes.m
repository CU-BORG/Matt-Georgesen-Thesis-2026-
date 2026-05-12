%% Generate Multi-Exponential Training Data - BATCH SAMPLE SIZES
% Generate datasets with different numbers of training images
% Saves to E drive for each sample size: 1, 10, 100, 200, 500, 1000
% ALL parameters (tau1, tau2, f1) drawn from UNIFORM distributions
% NO spatial correlation for any parameter
% @author: mg

clear; clc;

%% Tau ranges - similar to real data statistics
% Real data: tau1 [0.124, 0.784] ns, tau2 [1.528, 4.293] ns
TAU_1_MIN = 0.20;   % Narrower range for better learning
TAU_1_MAX = 0.70;
TAU_2_MIN = 1.20;   % Wider range than real data
TAU_2_MAX = 4.50;

%% Sample sizes to generate
SAMPLE_SIZES = [1, 10, 100, 200, 500, 1000];

%% Configuration
config.tau_Num = 2;              % Always 2 components
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins

% Path to 512x512 intensity images (PNG format)
config.image_source_path = 'C:\Users\mcg11923\Thesis\train';
config.image_pattern = '*.png';

% Tau ranges (uniform random)
config.tau_ranges = [TAU_1_MIN, TAU_1_MAX; TAU_2_MIN, TAU_2_MAX];

% Photon count scaling based on intensity
config.photon_scale_min = 500;    % Minimum photons for darkest pixels
config.photon_scale_max = 5000;   % Maximum photons for brightest pixels
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons

% Fraction range (uniform random)
config.f_range = [0.1, 0.9];

fprintf('=================================================================\n');
fprintf('Multi-Exponential Batch Generation - Different Sample Sizes\n');
fprintf('(FULLY FLAT RANDOM - NO SPATIAL CORRELATION)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Tau1 range: [%.2f, %.2f] ns (UNIFORM RANDOM)\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  Tau2 range: [%.2f, %.2f] ns (UNIFORM RANDOM)\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  Sample sizes to generate: %s\n', mat2str(SAMPLE_SIZES));
fprintf('  Output location: E drive\n');
fprintf('  Input image size: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output image size: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);
fprintf('  Bin width: %.4f ns\n', config.bin_width);
fprintf('  Photon range: [%d, %d]\n', config.photon_scale_min, config.photon_scale_max);
fprintf('  Image source: %s\n', config.image_source_path);
fprintf('\nDistribution Settings:\n');
fprintf('  Tau1: Uniform[%.2f, %.2f] ns (per pixel, independent)\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  Tau2: Uniform[%.2f, %.2f] ns (per pixel, independent)\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  Fraction f1: Uniform[%.2f, %.2f] (per pixel, independent)\n', config.f_range(1), config.f_range(2));
fprintf('  NO SPATIAL CORRELATION for any parameter\n');
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

%% Load 512x512 intensity images
fprintf('\nLoading 512x512 intensity images...\n');
image_files = dir(fullfile(config.image_source_path, config.image_pattern));

if isempty(image_files)
    error('No images found in %s with pattern %s', ...
        config.image_source_path, config.image_pattern);
end

n_available = length(image_files);
fprintf('Found %d image files\n', n_available);

% Check if we have enough images
max_needed = max(SAMPLE_SIZES);
if n_available < max_needed
    error('Not enough images! Need %d, but only %d available', max_needed, n_available);
end

% Shuffle images for random selection
shuffled_indices = randperm(n_available);
image_files = image_files(shuffled_indices);

%% MAIN LOOP: Generate for each sample size
overall_start = tic;

for size_idx = 1:length(SAMPLE_SIZES)
    N_images = SAMPLE_SIZES(size_idx);

    fprintf('\n=================================================================\n');
    fprintf('GENERATING DATASET %d/%d: N = %d images\n', size_idx, length(SAMPLE_SIZES), N_images);
    fprintf('=================================================================\n');

    % Create output directory on E drive
    config.output_dir = sprintf('E:\\fully_flatrandom_n%04d', N_images);

    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end

    fprintf('Output directory: %s\n', config.output_dir);
    fprintf('Progress: ');

    size_start = tic;

    %% Generate images for this sample size
    for img_idx = 1:N_images
        img_tic = tic;

        %% Load and process intensity image
        img_path = fullfile(image_files(img_idx).folder, image_files(img_idx).name);
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
        img_normalized = double(img_gray) / 255.0;

        % Downsample to 256x256 using bilinear interpolation
        img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
        Int = img_256;

        %% Generate FLAT RANDOM tau maps (NO spatial correlation)
        tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

        % Tau1: Uniform random [TAU_1_MIN, TAU_1_MAX]
        tau_map(:, :, 1) = TAU_1_MIN + (TAU_1_MAX - TAU_1_MIN) * rand(config.image_size_output, config.image_size_output);

        % Tau2: Uniform random [TAU_2_MIN, TAU_2_MAX]
        tau_map(:, :, 2) = TAU_2_MIN + (TAU_2_MAX - TAU_2_MIN) * rand(config.image_size_output, config.image_size_output);

        %% Generate flat random fraction maps
        f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

        F_MIN = config.f_range(1);
        F_MAX = config.f_range(2);

        % f1: Uniform random [F_MIN, F_MAX]
        f_map(:, :, 1) = F_MIN + (F_MAX - F_MIN) * rand(config.image_size_output, config.image_size_output);
        f_map(:, :, 2) = 1 - f_map(:, :, 1);

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
                tau_pixel = squeeze(tau_map(y, x, :))';  % Flat random
                f_pixel = squeeze(f_map(y, x, :))';      % Flat random

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
        output_filename = sprintf('Sample_%04d_fully_flatrandom.mat', img_idx);
        output_path = fullfile(config.output_dir, output_filename);

        save(output_path, 'Hist', 'tau_gt_components', 'f_gt_components', ...
             'tau_gt_avg', 'Int', 'max_components', '-v7.3');

        img_time = toc(img_tic);
        fprintf('%d(%.1fs) ', img_idx, img_time);

        if mod(img_idx, 10) == 0
            fprintf('\n          ');
        end
    end

    size_time = toc(size_start);
    fprintf('\n  Completed in %.1f seconds (%.2f images/min)\n', ...
        size_time, N_images / size_time * 60);

    %% Save dataset info file
    info_path = fullfile(config.output_dir, 'dataset_info.txt');
    fid = fopen(info_path, 'w');
    fprintf(fid, 'FULLY FLAT RANDOM DATASET INFO\n');
    fprintf(fid, '=================================================================\n\n');
    fprintf(fid, 'Number of images: %d\n', N_images);
    fprintf(fid, 'Image size: %dx%d pixels\n', config.image_size_output, config.image_size_output);
    fprintf(fid, 'Time bins: %d\n', config.bin_Num);
    fprintf(fid, 'Bin width: %.4f ns\n\n', config.bin_width);
    fprintf(fid, 'Parameter Distributions (all UNIFORM, NO spatial correlation):\n');
    fprintf(fid, '  Tau1: Uniform[%.2f, %.2f] ns\n', TAU_1_MIN, TAU_1_MAX);
    fprintf(fid, '  Tau2: Uniform[%.2f, %.2f] ns\n', TAU_2_MIN, TAU_2_MAX);
    fprintf(fid, '  Fraction f1: Uniform[%.2f, %.2f]\n\n', config.f_range(1), config.f_range(2));
    fprintf(fid, 'Key Features:\n');
    fprintf(fid, '  - Each pixel completely independent\n');
    fprintf(fid, '  - NO spatial correlation for any parameter\n');
    fprintf(fid, '  - Narrower tau1 range for better learning\n');
    fprintf(fid, '  - All histograms approximately uniform\n');
    fclose(fid);
end

%% Final summary
total_time = toc(overall_start);
fprintf('\n=================================================================\n');
fprintf('BATCH GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Generated %d datasets with different sample sizes\n', length(SAMPLE_SIZES));
fprintf('Sample sizes: %s\n', mat2str(SAMPLE_SIZES));
fprintf('Total runtime: %.1f seconds (%.2f minutes, %.2f hours)\n', ...
    total_time, total_time/60, total_time/3600);
fprintf('\nOutput directories on E drive:\n');
for size_idx = 1:length(SAMPLE_SIZES)
    fprintf('  E:\\fully_flatrandom_n%04d\\ (%d images)\n', ...
        SAMPLE_SIZES(size_idx), SAMPLE_SIZES(size_idx));
end
fprintf('\nParameter ranges:\n');
fprintf('  Tau1: Uniform[%.2f, %.2f] ns\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  Tau2: Uniform[%.2f, %.2f] ns\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  Fraction f1: Uniform[%.2f, %.2f]\n', config.f_range(1), config.f_range(2));
fprintf('=================================================================\n');
