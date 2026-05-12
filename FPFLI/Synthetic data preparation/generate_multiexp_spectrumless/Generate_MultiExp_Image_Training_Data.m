%% Generate Multi-Exponential Image Training Data (SPECTRUMLESS VERSION)
% Creates 256x256x256 training data cubes from 512x512 intensity images
% with FIXED lifetime values (0.4 ns and 4.5 ns) and NO spectral variation.
% Downsamples to 256x256 and generates multi-exponential decay for each pixel.
%
% Output: Training_data_multiexp_images_spectrumless.mat containing:
%   - Hist: [N_images, 256, 256, 256] - Fluorescence decay histograms
%   - Int_512: [N_images, 512, 512] - Original intensity images
%   - Int_256: [N_images, 256, 256] - Downsampled intensity images
%   - tau_components: [N_images, 256, 256, 2] - Fixed lifetime maps (always [0.4, 4.5])
%   - f_components: [N_images, 256, 256, 2] - Fraction maps (varying)
%   - tau_avg: [N_images, 256, 256] - Average lifetime maps
%
% Author: MG
% Date: 2025-12-07

clear; clc;

%% FIXED TAU VALUES (NO SPECTRAL VARIATION)
FIXED_TAU_1 = 0.4;   % Short lifetime component (ns)
FIXED_TAU_2 = 4.5;   % Long lifetime component (ns)

%% Configuration
config.tau_Num = 2;              % Always 2 components (fixed)
config.N_images = 100;           % Number of training images to generate
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins
config.output_dir = 'spectrumless_training_data';  % Directory for individual files

% Path to 512x512 intensity images (PNG format)
% Modify this path to point to your image directory
config.image_source_path = 'C:\Users\mcg11923\Thesis\train';  % Change this to your path
config.image_pattern = '*.png';               % Image file pattern

% Fixed lifetime values (NO VARIATION)
config.tau_fixed = [FIXED_TAU_1, FIXED_TAU_2];

% Photon count scaling based on intensity
% Each pixel's photon count will be: intensity * scale_factor
config.photon_scale_min = 50;    % Minimum photons for darkest pixels
config.photon_scale_max = 500;   % Maximum photons for brightest pixels
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons

fprintf('=================================================================\n');
fprintf('Multi-Exponential Image Training Data Generation (SPECTRUMLESS)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d (FIXED)\n', config.tau_Num);
fprintf('  Fixed tau values: [%.1f, %.1f] ns (NO VARIATION)\n', FIXED_TAU_1, FIXED_TAU_2);
fprintf('  Training images: %d\n', config.N_images);
fprintf('  Input image size: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output image size: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);
fprintf('  Bin width: %.4f ns\n', config.bin_width);
fprintf('  Photon range: [%d, %d]\n', config.photon_scale_min, config.photon_scale_max);
fprintf('  Image source: %s\n', config.image_source_path);
fprintf('  Output directory: %s\n', config.output_dir);
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
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
    fprintf('Created output directory: %s\n', config.output_dir);
else
    fprintf('Output directory already exists: %s\n', config.output_dir);
end

%% Generate training images
fprintf('\nGenerating %d multi-exponential image cubes with FIXED tau=[%.1f, %.1f] ns...\n', ...
    config.N_images, FIXED_TAU_1, FIXED_TAU_2);
fprintf('Progress: ');

total_tic = tic;

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
    Int_512 = double(img_gray);  % [512, 512]
    img_normalized = double(img_gray) / 255.0;

    % Downsample to 256x256 using bilinear interpolation
    img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
    Int = img_256;  % [256, 256] - renamed for consistency with training format

    %% Generate FIXED lifetime and VARYING fraction maps
    % Lifetime is CONSTANT across all pixels (no spatial variation)
    % Fractions vary spatially to model different mixing ratios
    tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);
    f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    % Generate smooth spatial variations in fractions using patches
    patch_size = 32; % Size of regions with constant fractions
    n_patches = config.image_size_output / patch_size;

    for py = 1:n_patches
        for px = 1:n_patches
            % FIXED lifetimes for all patches (NO VARIATION)
            tau_patch = config.tau_fixed;  % Always [0.4, 4.5]

            % Random fractions for this patch (Dirichlet distribution)
            % Using exponential random variables (no Statistics Toolbox needed)
            % Dirichlet(alpha) where alpha = [1,1,...] is equivalent to
            % normalizing independent Exp(1) random variables
            exp_rvs = -log(rand(1, config.tau_Num));  % Exponential(1) samples
            f_patch = exp_rvs / sum(exp_rvs);  % Normalize to sum to 1

            % Assign to pixels in this patch
            y_start = (py-1)*patch_size + 1;
            y_end = py*patch_size;
            x_start = (px-1)*patch_size + 1;
            x_end = px*patch_size;

            for comp = 1:config.tau_Num
                tau_map(y_start:y_end, x_start:x_end, comp) = tau_patch(comp);
                f_map(y_start:y_end, x_start:x_end, comp) = f_patch(comp);
            end
        end
    end

    % Store ground truth maps (for this single image)
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

            % Get FIXED lifetime components and VARYING fractions for this pixel
            tau_pixel = squeeze(tau_map(y, x, :))';  % Always [0.4, 4.5]
            f_pixel = squeeze(f_map(y, x, :))';      % Varies by patch

            % Generate multi-exponential decay
            decay = Fluorescence_multi_decay_nonhomopp(...
                config.tau_Num, tau_pixel, f_pixel, N_photons, config.bin_width, IRF);

            hist_cube(y, x, :) = decay;
        end
    end

    % Store histogram cube (for this single image)
    Hist = hist_cube;  % [256, 256, 256]

    % Store max_components for compatibility
    max_components = config.tau_Num;

    %% Save individual file
    output_filename = sprintf('Sample_%03d_spectrumless.mat', img_idx);
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
fprintf('\n\nData generation completed in %.1f seconds (%.2f images/min)\n', ...
    total_time, config.N_images / total_time * 60);

%% Load first file for statistics
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Loading first file for statistics...\n');
first_file = fullfile(config.output_dir, 'Sample_001_spectrumless.mat');
sample_data = load(first_file);

fprintf('\nData Statistics (based on Sample_001):\n');
fprintf('-----------------------------------------------------------------\n');

fprintf('Image Sizes:\n');
fprintf('  Input: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);

fprintf('\nLifetime Components (FIXED - NO VARIATION):\n');
for comp = 1:config.tau_Num
    tau_data = sample_data.tau_gt_components(:, :, comp);
    fprintf('  Component %d:\n', comp);
    fprintf('    Fixed value: %.3f ns\n', config.tau_fixed(comp));
    fprintf('    Range: [%.3f, %.3f] ns (should be constant)\n', min(tau_data(:)), max(tau_data(:)));
    fprintf('    Mean: %.3f ns, Std: %.6f ns (std should be ~0)\n', mean(tau_data(:)), std(tau_data(:)));
end

fprintf('\nFraction Components (VARYING):\n');
for comp = 1:config.tau_Num
    f_data = sample_data.f_gt_components(:, :, comp);
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f]\n', min(f_data(:)), max(f_data(:)));
    fprintf('    Mean: %.3f\n', mean(f_data(:)));
end

fprintf('\nAverage Lifetime:\n');
fprintf('  Range: [%.3f, %.3f] ns\n', min(sample_data.tau_gt_avg(:)), max(sample_data.tau_gt_avg(:)));
fprintf('  Mean: %.3f ns\n', mean(sample_data.tau_gt_avg(:)));
fprintf('  Note: Average lifetime varies only due to fraction variation\n');

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
fprintf('tau_gt_components   | [%d, %d, %d]     | Fixed lifetime maps ([%.1f, %.1f])\n', ...
    size(sample_data.tau_gt_components, 1), size(sample_data.tau_gt_components, 2), ...
    size(sample_data.tau_gt_components, 3), FIXED_TAU_1, FIXED_TAU_2);
fprintf('f_gt_components     | [%d, %d, %d]     | Fraction maps (varying)\n', ...
    size(sample_data.f_gt_components, 1), size(sample_data.f_gt_components, 2), ...
    size(sample_data.f_gt_components, 3));
fprintf('tau_gt_avg          | [%d, %d]      | Average lifetime map\n', ...
    size(sample_data.tau_gt_avg, 1), size(sample_data.tau_gt_avg, 2));
fprintf('Int                 | [%d, %d]      | Intensity image (256x256)\n', ...
    size(sample_data.Int, 1), size(sample_data.Int, 2));
fprintf('max_components      | scalar                | Number of components (%d)\n', ...
    sample_data.max_components);
fprintf('-----------------------------------------------------------------\n');

%% Create visualization
fprintf('\nCreating visualization...\n');

figure('Position', [100, 100, 1400, 900]);

% Downsampled 256x256
subplot(3, 4, 1);
imagesc(sample_data.Int);
colorbar;
title('256x256 Intensity');
axis image;

% Lifetime component 1 (should be constant at 0.4)
subplot(3, 4, 2);
imagesc(sample_data.tau_gt_components(:, :, 1));
colorbar;
title(sprintf('Lifetime Component 1 (FIXED: %.1f ns)', FIXED_TAU_1));
axis image;

% Lifetime component 2 (should be constant at 4.5)
subplot(3, 4, 3);
imagesc(sample_data.tau_gt_components(:, :, 2));
colorbar;
title(sprintf('Lifetime Component 2 (FIXED: %.1f ns)', FIXED_TAU_2));
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

% Lifetime histogram (should show two peaks at 0.4 and 4.5)
subplot(3, 4, 9);
tau_all = sample_data.tau_gt_components(:);
histogram(tau_all, 50);
xlabel('Lifetime (ns)');
ylabel('Count');
title('Lifetime Distribution (FIXED)');
grid on;

% Fraction histogram
subplot(3, 4, 10);
f_all = sample_data.f_gt_components(:);
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
% Load 3 random samples to compare
sample_indices = [1, 50, 100];
for i = 1:length(sample_indices)
    sample_file = fullfile(config.output_dir, sprintf('Sample_%03d_spectrumless.mat', sample_indices(i)));
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

sgtitle(sprintf('Multi-Exponential Image Data (SPECTRUMLESS: tau=[%.1f, %.1f] ns, %d samples)', ...
    FIXED_TAU_1, FIXED_TAU_2, config.N_images), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
fig_name = fullfile(config.output_dir, 'spectrumless_visualization.png');
saveas(gcf, fig_name);
fprintf('Visualization saved to %s\n', fig_name);

%% Final summary
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (SPECTRUMLESS)!\n');
fprintf('=================================================================\n');
fprintf('Generated %d training images:\n', config.N_images);
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes per file\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('  - 2 FIXED lifetime components: [%.1f, %.1f] ns\n', FIXED_TAU_1, FIXED_TAU_2);
fprintf('  - Varying fraction components per pixel\n');
fprintf('\nOutput directory: %s\n', config.output_dir);
fprintf('Files saved: Sample_001_spectrumless.mat through Sample_%03d_spectrumless.mat\n', config.N_images);
fprintf('Approx file size per image: %.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5); % Approx with metadata
fprintf('Total size: ~%.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * config.N_images);
fprintf('=================================================================\n');
