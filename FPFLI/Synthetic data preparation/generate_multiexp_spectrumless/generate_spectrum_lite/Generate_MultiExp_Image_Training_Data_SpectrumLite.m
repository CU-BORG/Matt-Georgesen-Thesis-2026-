%% Generate Multi-Exponential Image Training Data (SPECTRUM-LITE VERSION)
% Creates 256x256x256 training data cubes from 512x512 intensity images
% with GAUSSIAN-distributed lifetime values around central values (0.4 ns and 4.5 ns).
% Lifetimes vary slightly with Gaussian noise to simulate "spectrum-lite" variation.
%
% Output: Individual .mat files containing:
%   - Hist: [256, 256, 256] - Fluorescence decay histograms
%   - tau_gt_components: [256, 256, 2] - Gaussian-distributed lifetime maps
%   - f_gt_components: [256, 256, 2] - Fraction maps (varying)
%   - tau_gt_avg: [256, 256] - Average lifetime maps
%   - Int: [256, 256] - Intensity image
%   - max_components: scalar - Number of components (2)
%
% Author: MG
% Date: 2025-12-08

clear; clc;

%% CENTRAL TAU VALUES WITH GAUSSIAN VARIATION
TAU_1_CENTER = 1.0;   % Short lifetime center (ns)
TAU_2_CENTER = 4.0;   % Long lifetime center (ns)

% Define 10 standard deviation levels (increasing spectral variation)
% Progression from minimal variation (spectrumless-like) to full overlap
STD_LEVELS = 10;
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05];
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50];

%% Configuration
config.tau_Num = 2;              % Always 2 components
config.N_images = 400;           % Number of training images per level
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins

% Path to 512x512 intensity images (PNG format)
config.image_source_path = 'C:\Users\mcg11923\Thesis\train';  % Change this to your path
config.image_pattern = '*.png';

% Tau parameters for Gaussian distributions
config.tau_center = [TAU_1_CENTER, TAU_2_CENTER];

% Photon count scaling based on intensity
config.photon_scale_min = 500;    % Minimum photons for darkest pixels (10x increase)
config.photon_scale_max = 5000;   % Maximum photons for brightest pixels (10x increase)
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons

fprintf('=================================================================\n');
fprintf('Multi-Exponential Image Training Data Generation (SPECTRUM PROGRESSION)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Tau centers: %.2f ns, %.2f ns\n', TAU_1_CENTER, TAU_2_CENTER);
fprintf('  Standard deviation levels: %d (from minimal to full overlap)\n', STD_LEVELS);
fprintf('  Training images per level: %d\n', config.N_images);
fprintf('  Total images: %d\n', STD_LEVELS * config.N_images);
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

%% Main loop: Generate data for each standard deviation level
fprintf('\n=================================================================\n');
fprintf('Starting generation for %d standard deviation levels...\n', STD_LEVELS);
fprintf('=================================================================\n\n');

total_tic = tic;

for std_level = 1:STD_LEVELS
    %% Set current standard deviation values
    TAU_1_STD = TAU_1_STD_ARRAY(std_level);
    TAU_2_STD = TAU_2_STD_ARRAY(std_level);
    config.tau_std = [TAU_1_STD, TAU_2_STD];

    %% Create level-specific output directory
    config.output_dir = sprintf('spectrum_level_%02d_std_%.2f_%.2f', ...
        std_level, TAU_1_STD, TAU_2_STD);

    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end

    fprintf('-----------------------------------------------------------------\n');
    fprintf('LEVEL %d/%d: Tau1_std=%.2f ns, Tau2_std=%.2f ns\n', ...
        std_level, STD_LEVELS, TAU_1_STD, TAU_2_STD);
    fprintf('Output: %s\n', config.output_dir);
    fprintf('Progress: ');

    level_tic = tic;

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
    Int_512 = double(img_gray);
    img_normalized = double(img_gray) / 255.0;

    % Downsample to 256x256 using bilinear interpolation
    img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
    Int = img_256;

    %% Generate GAUSSIAN-distributed lifetime and VARYING fraction maps
    % Each pixel gets independently sampled tau and fraction values
    % Lifetimes: Gaussian distributions around central values
    % Fractions: Random per-pixel using Dirichlet distribution
    tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);
    f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    % Generate per-pixel variations (true spectrum-lite with pixel-level diversity)
    for y = 1:config.image_size_output
        for x = 1:config.image_size_output
            % Generate tau values from Gaussian distributions for this pixel
            % Ensure positive values by clamping to reasonable bounds
            for comp = 1:config.tau_Num
                % Sample from Gaussian distribution
                tau_sample = config.tau_center(comp) + config.tau_std(comp) * randn();

                % Clamp to reasonable bounds (positive and not too extreme)
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
            tau_pixel = squeeze(tau_map(y, x, :))';  % Gaussian-distributed
            f_pixel = squeeze(f_map(y, x, :))';      % Varies by patch

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
    output_filename = sprintf('Sample_%03d_spectrumspectrum.mat', img_idx);
    output_path = fullfile(config.output_dir, output_filename);

    save(output_path, 'Hist', 'tau_gt_components', 'f_gt_components', ...
         'tau_gt_avg', 'Int', 'max_components', '-v7.3');

    img_time = toc(img_tic);
    fprintf('%d (%.1fs) ', img_idx, img_time);

    if mod(img_idx, 10) == 0
        fprintf('\n          ');
    end
end

level_time = toc(level_tic);
fprintf('\n  Level %d completed in %.1f seconds (%.2f images/min)\n', ...
    std_level, level_time, config.N_images / level_time * 60);

%% Load first file for statistics (per level)
first_file = fullfile(config.output_dir, 'Sample_001_spectrumspectrum.mat');
sample_data = load(first_file);

fprintf('\nData Statistics (based on Sample_001):\n');
fprintf('-----------------------------------------------------------------\n');

fprintf('Image Sizes:\n');
fprintf('  Input: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);

fprintf('\nLifetime Components (GAUSSIAN-DISTRIBUTED):\n');
for comp = 1:config.tau_Num
    tau_data = sample_data.tau_gt_components(:, :, comp);
    tau_data_nonzero = tau_data(tau_data > 1e-6);
    fprintf('  Component %d:\n', comp);
    fprintf('    Target: Gaussian(μ=%.2f ns, σ=%.3f ns)\n', ...
        config.tau_center(comp), config.tau_std(comp));
    fprintf('    Observed range: [%.3f, %.3f] ns\n', min(tau_data_nonzero), max(tau_data_nonzero));
    fprintf('    Observed mean: %.3f ns, Std: %.3f ns\n', mean(tau_data_nonzero), std(tau_data_nonzero));
end

fprintf('\nFraction Components (VARYING):\n');
for comp = 1:config.tau_Num
    f_data = sample_data.f_gt_components(:, :, comp);
    f_data_nonzero = f_data(f_data > 1e-6);
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f]\n', min(f_data_nonzero), max(f_data_nonzero));
    fprintf('    Mean: %.3f\n', mean(f_data_nonzero));
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
fprintf('tau_gt_components   | [%d, %d, %d]     | Gaussian-distributed lifetimes\n', ...
    size(sample_data.tau_gt_components, 1), size(sample_data.tau_gt_components, 2), ...
    size(sample_data.tau_gt_components, 3));
fprintf('f_gt_components     | [%d, %d, %d]     | Fraction maps (varying)\n', ...
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

% Lifetime component 1 (Gaussian distributed)
subplot(3, 4, 2);
imagesc(sample_data.tau_gt_components(:, :, 1));
colorbar;
title(sprintf('Tau1 (Gaussian: μ=%.2f, σ=%.3f ns)', TAU_1_CENTER, TAU_1_STD));
axis image;

% Lifetime component 2 (Gaussian distributed)
subplot(3, 4, 3);
imagesc(sample_data.tau_gt_components(:, :, 2));
colorbar;
title(sprintf('Tau2 (Gaussian: μ=%.2f, σ=%.3f ns)', TAU_2_CENTER, TAU_2_STD));
axis image;

% Average lifetime
subplot(3, 4, 4);
imagesc(sample_data.tau_gt_avg);
colorbar;
title('Average Lifetime (ns)');
axis image;

% Fraction component 1
subplot(3, 4, 5);
imagesc(sample_data.f_gt_components(:, :, 1));
colorbar;
title('Fraction Component 1 (VARYING)');
axis image;
caxis([0, 1]);

% Fraction component 2
subplot(3, 4, 6);
imagesc(sample_data.f_gt_components(:, :, 2));
colorbar;
title('Fraction Component 2 (VARYING)');
axis image;
caxis([0, 1]);

% Total photon map
subplot(3, 4, 7);
photon_map = sum(sample_data.Hist, 3);
imagesc(photon_map);
colorbar;
title('Total Photons per Pixel');
axis image;

% Sample decay curves
subplot(3, 4, 8);
y_sample = [64, 128, 192];
x_sample = [64, 128, 192];
for i = 1:length(y_sample)
    decay_curve = squeeze(sample_data.Hist(y_sample(i), x_sample(i), :));
    plot(decay_curve, 'LineWidth', 1.5, 'DisplayName', ...
        sprintf('(%d,%d)', y_sample(i), x_sample(i)));
    hold on;
end
xlabel('Time Bin');
ylabel('Photon Count');
title('Sample Decay Curves');
legend('Location', 'best');
grid on;

% Lifetime histogram (should show Gaussian distributions)
subplot(3, 4, 9);
tau_all = sample_data.tau_gt_components(:);
tau_all = tau_all(tau_all > 1e-6);  % Remove zeros
histogram(tau_all, 50);
xlabel('Lifetime (ns)');
ylabel('Count');
title('Lifetime Distribution (Gaussian)');
grid on;

% Fraction histogram
subplot(3, 4, 10);
f_all = sample_data.f_gt_components(:);
f_all = f_all(f_all > 1e-6);  % Remove zeros
histogram(f_all, 50);
xlabel('Fraction');
ylabel('Count');
title('Fraction Distribution (VARYING)');
grid on;

% Intensity vs photons
subplot(3, 4, 11);
int_flat = sample_data.Int(:);
photon_flat = photon_map(:);
scatter(int_flat, photon_flat, 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Normalized Intensity');
ylabel('Total Photons');
title('Intensity vs Photon Count');
grid on;

% Decay curve comparison across samples
subplot(3, 4, 12);
sample_indices = [1, 50, 100];
for i = 1:length(sample_indices)
    sample_file = fullfile(config.output_dir, sprintf('Sample_%03d_spectrumspectrum.mat', sample_indices(i)));
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

sgtitle(sprintf('Spectrum Level %d (Gaussian τ1=%.2f±%.3f, τ2=%.2f±%.3f ns)', ...
    std_level, TAU_1_CENTER, TAU_1_STD, TAU_2_CENTER, TAU_2_STD), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
fig_name = fullfile(config.output_dir, sprintf('spectrum_level_%02d_visualization.png', std_level));
saveas(gcf, fig_name);
close(gcf);
fprintf('  Visualization saved to %s\n\n', fig_name);

end  % End of std_level loop

%% Final summary
total_time = toc(total_tic);
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (SPECTRUM PROGRESSION)!\n');
fprintf('=================================================================\n');
fprintf('Generated %d standard deviation levels:\n', STD_LEVELS);
fprintf('  - Tau centers: %.2f ns, %.2f ns\n', TAU_1_CENTER, TAU_2_CENTER);
fprintf('  - Std dev range: [%.2f-%.2f ns] for tau1, [%.2f-%.2f ns] for tau2\n', ...
    TAU_1_STD_ARRAY(1), TAU_1_STD_ARRAY(end), TAU_2_STD_ARRAY(1), TAU_2_STD_ARRAY(end));
fprintf('  - Images per level: %d\n', config.N_images);
fprintf('  - Total images: %d\n', STD_LEVELS * config.N_images);
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes per file\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('  - Varying fraction components per pixel\n');
fprintf('\nOutput directories created:\n');
for i = 1:STD_LEVELS
    dir_name = sprintf('  spectrum_level_%02d_std_%.2f_%.2f/', i, TAU_1_STD_ARRAY(i), TAU_2_STD_ARRAY(i));
    fprintf('%s\n', dir_name);
end
fprintf('\nFiles per directory: Sample_001_spectrumspectrum.mat through Sample_%03d_spectrumspectrum.mat\n', config.N_images);
fprintf('Approx file size per image: %.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5);
fprintf('Total size: ~%.1f MB (~%.1f GB)\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * STD_LEVELS * config.N_images, ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * STD_LEVELS * config.N_images / 1024);
fprintf('Total runtime: %.1f seconds (%.2f minutes, %.2f hours)\n', ...
    total_time, total_time/60, total_time/3600);
fprintf('=================================================================\n');
