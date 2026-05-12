%% Generate Multi-Exponential Training Data (FULLY FLAT RANDOM)
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

%% Configuration
config.tau_Num = 2;              % Always 2 components
config.N_images = 400;           % Number of training images
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
fprintf('Multi-Exponential Image Training Data Generation\n');
fprintf('(FULLY FLAT RANDOM - NO SPATIAL CORRELATION)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Tau1 range: [%.2f, %.2f] ns (UNIFORM RANDOM)\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  Tau2 range: [%.2f, %.2f] ns (UNIFORM RANDOM)\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  Training images: %d\n', config.N_images);
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

% Limit to requested number of images
n_available = length(image_files);
if n_available < config.N_images
    warning('Only %d images available, using all of them', n_available);
    config.N_images = n_available;
end

% Select random subset if more images available than needed
if n_available > config.N_images
    selected_indices = randperm(n_available, config.N_images);
    image_files = image_files(selected_indices);
end

fprintf('Loaded %d image files\n', config.N_images);

%% Create output directory
config.output_dir = 'E:\fully_flatrandom';

if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('\n=================================================================\n');
fprintf('Starting generation of %d images with FULLY FLAT RANDOM parameters...\n', config.N_images);
fprintf('Output directory: %s\n', config.output_dir);
fprintf('=================================================================\n\n');

total_tic = tic;
fprintf('Progress: ');

%% Main loop: Generate data for all images
for img_idx = 1:config.N_images
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
    output_filename = sprintf('Sample_%03d_fully_flatrandom.mat', img_idx);
    output_path = fullfile(config.output_dir, output_filename);

    save(output_path, 'Hist', 'tau_gt_components', 'f_gt_components', ...
         'tau_gt_avg', 'Int', 'max_components', '-v7.3');

    img_time = toc(img_tic);
    fprintf('%d (%.1fs) ', img_idx, img_time);

    if mod(img_idx, 10) == 0
        fprintf('\n          ');
    end
end

total_time = toc(total_tic);
fprintf('\n\nGeneration completed in %.1f seconds (%.2f images/min)\n', ...
    total_time, config.N_images / total_time * 60);

%% Load first file for statistics
first_file = fullfile(config.output_dir, 'Sample_001_fully_flatrandom.mat');
sample_data = load(first_file);

fprintf('\n=================================================================\n');
fprintf('Data Statistics (based on Sample_001):\n');
fprintf('=================================================================\n');

fprintf('\nImage Sizes:\n');
fprintf('  Input: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);

fprintf('\nLifetime Components (FLAT RANDOM - NO CORRELATION):\n');
for comp = 1:config.tau_Num
    tau_data = sample_data.tau_gt_components(:, :, comp);
    tau_data_nonzero = tau_data(tau_data > 1e-6);
    fprintf('  Component %d:\n', comp);
    fprintf('    Target: Uniform[%.2f, %.2f] ns\n', ...
        config.tau_ranges(comp, 1), config.tau_ranges(comp, 2));
    fprintf('    Observed range: [%.3f, %.3f] ns\n', min(tau_data_nonzero), max(tau_data_nonzero));
    fprintf('    Observed mean: %.3f ns, Std: %.3f ns\n', mean(tau_data_nonzero), std(tau_data_nonzero));
end

fprintf('\nFraction Components (FLAT RANDOM):\n');
for comp = 1:config.tau_Num
    f_data = sample_data.f_gt_components(:, :, comp);
    f_data_nonzero = f_data(f_data > 1e-6);
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f]\n', min(f_data_nonzero), max(f_data_nonzero));
    fprintf('    Mean: %.3f\n', mean(f_data_nonzero));
    fprintf('    Std:  %.3f\n', std(f_data_nonzero));
end

fprintf('\nAverage Lifetime:\n');
tau_avg_nonzero = sample_data.tau_gt_avg(sample_data.tau_gt_avg > 1e-6);
fprintf('  Range: [%.3f, %.3f] ns\n', min(tau_avg_nonzero), max(tau_avg_nonzero));
fprintf('  Mean: %.3f ns\n', mean(tau_avg_nonzero));
fprintf('  Note: Average lifetime varies due to both tau and fraction variation\n');

fprintf('\nPhoton Counts (Sample_001):\n');
total_photons = sum(sample_data.Hist(:));
fprintf('  Total photons: %.0f\n', total_photons);

%% Display data structure per file
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Output Data Structure (per .mat file):\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Variable Name       | Size                  | Description\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Hist                | [%d, %d, %d]  | Decay histograms\n', ...
    size(sample_data.Hist, 1), size(sample_data.Hist, 2), size(sample_data.Hist, 3));
fprintf('tau_gt_components   | [%d, %d, %d]     | Flat random lifetimes\n', ...
    size(sample_data.tau_gt_components, 1), size(sample_data.tau_gt_components, 2), ...
    size(sample_data.tau_gt_components, 3));
fprintf('f_gt_components     | [%d, %d, %d]     | Flat random fractions\n', ...
    size(sample_data.f_gt_components, 1), size(sample_data.f_gt_components, 2), ...
    size(sample_data.f_gt_components, 3));
fprintf('tau_gt_avg          | [%d, %d]      | Average lifetime map\n', ...
    size(sample_data.tau_gt_avg, 1), size(sample_data.tau_gt_avg, 2));
fprintf('Int                 | [%d, %d]      | Intensity image\n', ...
    size(sample_data.Int, 1), size(sample_data.Int, 2));
fprintf('max_components      | scalar                | Number of components (%d)\n', ...
    sample_data.max_components);
fprintf('-----------------------------------------------------------------\n');

%% Create visualization
fprintf('\nCreating visualization...\n');

figure('Position', [100, 100, 1400, 900]);

% Intensity
subplot(3, 4, 1);
imagesc(sample_data.Int);
colorbar;
title('256x256 Intensity');
axis image;

% Lifetime component 1 (Flat random)
subplot(3, 4, 2);
imagesc(sample_data.tau_gt_components(:, :, 1));
colorbar;
title(sprintf('Tau1 (Uniform [%.2f, %.2f] ns)', TAU_1_MIN, TAU_1_MAX));
axis image;

% Lifetime component 2 (Flat random)
subplot(3, 4, 3);
imagesc(sample_data.tau_gt_components(:, :, 2));
colorbar;
title(sprintf('Tau2 (Uniform [%.2f, %.2f] ns)', TAU_2_MIN, TAU_2_MAX));
axis image;

% Average lifetime
subplot(3, 4, 4);
imagesc(sample_data.tau_gt_avg);
colorbar;
title('Average Lifetime (ns)');
axis image;

% Fraction component 1 - should show NO spatial pattern (noise-like)
subplot(3, 4, 5);
imagesc(sample_data.f_gt_components(:, :, 1));
colorbar;
title(sprintf('Bound Fraction (Comp 1) [%.2f-%.2f] FLAT RANDOM', F_MIN, F_MAX));
axis image;
caxis([0, 1]);

% Fraction component 2
subplot(3, 4, 6);
imagesc(sample_data.f_gt_components(:, :, 2));
colorbar;
title(sprintf('Free Fraction (Comp 2) [%.2f-%.2f]', 1-F_MAX, 1-F_MIN));
axis image;
caxis([0, 1]);

% Total photon map
subplot(3, 4, 7);
photon_map = sum(sample_data.Hist, 3);
imagesc(photon_map);
colorbar;
title('Total Photons per Pixel');
axis image;

% Sample decay curves - pick pixels with very different bound fractions
subplot(3, 4, 8);
f1_map = sample_data.f_gt_components(:, :, 1);
[~, idx_min] = min(f1_map(:));
[~, idx_max] = max(f1_map(:));
[y_min, x_min] = ind2sub([256 256], idx_min);
[y_max, x_max] = ind2sub([256 256], idx_max);
y_mid = 128; x_mid = 128;

sample_coords = [y_min x_min; y_mid x_mid; y_max x_max];
labels = {'Min bound frac', 'Center', 'Max bound frac'};
for i = 1:3
    decay_curve = squeeze(sample_data.Hist(sample_coords(i,1), sample_coords(i,2), :));
    plot(decay_curve, 'LineWidth', 1.5, 'DisplayName', ...
        sprintf('%s (f1=%.2f)', labels{i}, f1_map(sample_coords(i,1), sample_coords(i,2))));
    hold on;
end
xlabel('Time Bin');
ylabel('Photon Count');
title('Decays: Min vs Max Bound Fraction');
legend('Location', 'best');
grid on;

% Fraction histogram - should show UNIFORM distribution
subplot(3, 4, 9);
f1_all = sample_data.f_gt_components(:, :, 1);
f1_all = f1_all(f1_all > 1e-6);
histogram(f1_all, 50);
xlabel('Bound Fraction (Component 1)');
ylabel('Count');
title('Bound Fraction Distribution (UNIFORM)');
grid on;

% Tau1 histogram - should show UNIFORM distribution
subplot(3, 4, 10);
tau1_all = sample_data.tau_gt_components(:, :, 1);
tau1_all = tau1_all(tau1_all > 1e-6);
histogram(tau1_all, 50);
xlabel('Tau1 (ns)');
ylabel('Count');
title('Tau1 Distribution (UNIFORM)');
grid on;

% Tau2 histogram - should show UNIFORM distribution
subplot(3, 4, 11);
tau2_all = sample_data.tau_gt_components(:, :, 2);
tau2_all = tau2_all(tau2_all > 1e-6);
histogram(tau2_all, 50);
xlabel('Tau2 (ns)');
ylabel('Count');
title('Tau2 Distribution (UNIFORM)');
grid on;

% Decay curve comparison across samples
subplot(3, 4, 12);
sample_indices = [1, 50, 100];
for i = 1:length(sample_indices)
    sample_file = fullfile(config.output_dir, sprintf('Sample_%03d_fully_flatrandom.mat', sample_indices(i)));
    if exist(sample_file, 'file')
        temp_data = load(sample_file);
        decay_curve = squeeze(temp_data.Hist(128, 128, :));
        plot(decay_curve, 'LineWidth', 1.5, 'DisplayName', sprintf('Sample %d', sample_indices(i)));
        hold on;
    end
end
xlabel('Time Bin');
ylabel('Photon Count');
title('Decay Comparison (pixel 128,128)');
legend('Location', 'best');
grid on;

sgtitle(sprintf('FULLY FLAT RANDOM (tau1=[%.2f,%.2f], tau2=[%.2f,%.2f] ns, f1=[%.2f,%.2f] all uniform)', ...
    TAU_1_MIN, TAU_1_MAX, TAU_2_MIN, TAU_2_MAX, F_MIN, F_MAX), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
fig_name = fullfile(config.output_dir, 'fully_flatrandom_visualization.png');
try
    saveas(gcf, fig_name);
    fprintf('  Visualization saved to %s\n\n', fig_name);
catch ME
    fprintf('  Warning: Could not save visualization: %s\n\n', ME.message);
end
close(gcf);

%% Final summary
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (FULLY FLAT RANDOM)!\n');
fprintf('=================================================================\n');
fprintf('Generated %d images with FULLY FLAT RANDOM parameters:\n', config.N_images);
fprintf('  - Tau1: Uniform[%.2f, %.2f] ns (wider than real data [0.12, 0.78])\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  - Tau2: Uniform[%.2f, %.2f] ns (wider than real data [1.53, 4.29])\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  - Fraction f1: Uniform[%.2f, %.2f] (much wider than real data ~[0.52, 0.90])\n', F_MIN, F_MAX);
fprintf('  - NO spatial correlation for any parameter\n');
fprintf('  - Each pixel completely independent\n');
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes per file\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('\nOutput directory: %s/\n', config.output_dir);
fprintf('Files: Sample_001_fully_flatrandom.mat through Sample_%03d_fully_flatrandom.mat\n', config.N_images);
fprintf('Approx file size per image: %.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5);
fprintf('Total size: ~%.1f MB (~%.1f GB)\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * config.N_images, ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * config.N_images / 1024);
fprintf('Total runtime: %.1f seconds (%.2f minutes, %.2f hours)\n', ...
    total_time, total_time/60, total_time/3600);
