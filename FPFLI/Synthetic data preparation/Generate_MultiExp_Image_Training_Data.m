%% Generate Multi-Exponential Image Training Data
% Creates 256x256x256 training data cubes from 512x512 intensity images
% Downsamples to 256x256 and generates multi-exponential decay for each pixel
%
% Output: Training_data_multiexp_images.mat containing:
%   - Hist: [N_images, 256, 256, 256] - Fluorescence decay histograms
%   - Int_512: [N_images, 512, 512] - Original intensity images
%   - Int_256: [N_images, 256, 256] - Downsampled intensity images
%   - tau_components: [N_images, 256, 256, n_components] - Lifetime maps
%   - f_components: [N_images, 256, 256, n_components] - Fraction maps
%   - tau_avg: [N_images, 256, 256] - Average lifetime maps
%
% Author: MG
% Date: 2025-10-07

clear; clc;

%% Configuration
config.tau_Num = 2;              % Number of lifetime components (2 or 3)
config.N_images = 100;           % Number of training images to generate
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins
config.output_file = 'Training_data_multiexp_images.mat';

% Path to 512x512 intensity images (PNG format)
% Modify this path to point to your image directory
config.image_source_path = 'E:\HPA_images\';  % Change this to your path
config.image_pattern = '*.png';               % Image file pattern

% Lifetime ranges for each component (in nanoseconds)
config.tau_ranges = [
    0.4, 2.5;   % Component 1 range (short)
    2.5, 5.0;   % Component 2 range (long)
    5.0, 8.0    % Component 3 range (very long) - only used if tau_Num=3
];

% Photon count scaling based on intensity
% Each pixel's photon count will be: intensity * scale_factor
config.photon_scale_min = 50;    % Minimum photons for darkest pixels
config.photon_scale_max = 500;   % Maximum photons for brightest pixels
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons


fprintf('Multi-Exponential Image Training Data Generation\n');

fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Training images: %d\n', config.N_images);
fprintf('  Input image size: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output image size: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);
fprintf('  Bin width: %.4f ns\n', config.bin_width);
fprintf('  Photon range: [%d, %d]\n', config.photon_scale_min, config.photon_scale_max);
fprintf('  Image source: %s\n', config.image_source_path);
fprintf('  Output file: %s\n', config.output_file);
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

%% Initialize output arrays
Hist = zeros(config.N_images, config.image_size_output, config.image_size_output, config.bin_Num);
Int_512 = zeros(config.N_images, config.image_size_input, config.image_size_input);
Int_256 = zeros(config.N_images, config.image_size_output, config.image_size_output);
tau_components = zeros(config.N_images, config.image_size_output, config.image_size_output, config.tau_Num);
f_components = zeros(config.N_images, config.image_size_output, config.image_size_output, config.tau_Num);
tau_avg = zeros(config.N_images, config.image_size_output, config.image_size_output);

%% Generate training images
fprintf('\nGenerating %d multi-exponential image cubes...\n', config.N_images);
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
    Int_512(img_idx, :, :) = double(img_gray);
    img_normalized = double(img_gray) / 255.0;

    % Downsample to 256x256 using bilinear interpolation
    img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
    Int_256(img_idx, :, :) = img_256;

    %% Generate lifetime and fraction maps (spatially varying)
    % Each pixel gets random lifetime components and fractions
    tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);
    f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    % Generate smooth spatial variations using Perlin noise or random patches
    % For simplicity, use piecewise constant regions (adjust as needed)
    patch_size = 32; % Size of regions with constant lifetime
    n_patches = config.image_size_output / patch_size;

    for py = 1:n_patches
        for px = 1:n_patches
            % Random lifetimes for this patch
            tau_patch = zeros(1, config.tau_Num);
            for comp = 1:config.tau_Num
                tau_min = config.tau_ranges(comp, 1);
                tau_max = config.tau_ranges(comp, 2);
                tau_patch(comp) = tau_min + rand() * (tau_max - tau_min);
            end

            % Random fractions (Dirichlet distribution)
            alpha = ones(1, config.tau_Num);
            f_patch = gamrnd(alpha, 1);
            f_patch = f_patch / sum(f_patch);

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

    % Store ground truth maps
    tau_components(img_idx, :, :, :) = tau_map;
    f_components(img_idx, :, :, :) = f_map;

    % Calculate average lifetime map
    for comp = 1:config.tau_Num
        tau_avg(img_idx, :, :) = squeeze(tau_avg(img_idx, :, :)) + ...
            squeeze(tau_map(:, :, comp)) .* squeeze(f_map(:, :, comp));
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
    Hist(img_idx, :, :, :) = hist_cube;

    img_time = toc(img_tic);
    fprintf('%d (%.1fs) ', img_idx, img_time);

    if mod(img_idx, 10) == 0
        fprintf('\n          ');
    end
end

total_time = toc(total_tic);
fprintf('\n\nData generation completed in %.1f seconds (%.2f images/min)\n', ...
    total_time, config.N_images / total_time * 60);

%% Compute and display statistics
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Data Statistics:\n');
fprintf('-----------------------------------------------------------------\n');

fprintf('Image Sizes:\n');
fprintf('  Input: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);

fprintf('\nLifetime Components (across all pixels):\n');
for comp = 1:config.tau_Num
    tau_data = tau_components(:, :, :, comp);
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f] ns\n', min(tau_data(:)), max(tau_data(:)));
    fprintf('    Mean: %.3f ns\n', mean(tau_data(:)));
end

fprintf('\nFraction Components (across all pixels):\n');
for comp = 1:config.tau_Num
    f_data = f_components(:, :, :, comp);
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f]\n', min(f_data(:)), max(f_data(:)));
    fprintf('    Mean: %.3f\n', mean(f_data(:)));
end

fprintf('\nAverage Lifetime:\n');
fprintf('  Range: [%.3f, %.3f] ns\n', min(tau_avg(:)), max(tau_avg(:)));
fprintf('  Mean: %.3f ns\n', mean(tau_avg(:)));

fprintf('\nPhoton Counts:\n');
total_photons = squeeze(sum(sum(Hist, 4), 3));
fprintf('  Total photons per image: %.0f +/- %.0f\n', ...
    mean(total_photons(:)), std(total_photons(:)));

%% Verify fraction sums
fraction_sums = sum(f_components, 4);
fprintf('\nFraction Sum Verification:\n');
fprintf('  Min: %.6f, Max: %.6f\n', min(fraction_sums(:)), max(fraction_sums(:)));
fprintf('  Mean: %.6f (should be 1.000000)\n', mean(fraction_sums(:)));
if max(abs(fraction_sums(:) - 1.0)) > 1e-6
    warning('Fraction sums deviate from 1.0!');
end

%% Save training data
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Saving training data to %s...\n', config.output_file);

save(config.output_file, 'Hist', 'Int_512', 'Int_256', ...
     'tau_components', 'f_components', 'tau_avg', 'IRF', 'config', '-v7.3');

fprintf('Training data saved successfully!\n');

%% Display data structure
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Output Data Structure:\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Variable Name       | Size                          | Description\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Hist                | [%d, %d, %d, %d]  | Decay histograms\n', ...
    size(Hist, 1), size(Hist, 2), size(Hist, 3), size(Hist, 4));
fprintf('Int_512             | [%d, %d, %d]      | Original 512x512 images\n', ...
    size(Int_512, 1), size(Int_512, 2), size(Int_512, 3));
fprintf('Int_256             | [%d, %d, %d]      | Downsampled 256x256 images\n', ...
    size(Int_256, 1), size(Int_256, 2), size(Int_256, 3));
fprintf('tau_components      | [%d, %d, %d, %d]     | Lifetime maps\n', ...
    size(tau_components, 1), size(tau_components, 2), size(tau_components, 3), size(tau_components, 4));
fprintf('f_components        | [%d, %d, %d, %d]     | Fraction maps\n', ...
    size(f_components, 1), size(f_components, 2), size(f_components, 3), size(f_components, 4));
fprintf('tau_avg             | [%d, %d, %d]      | Average lifetime maps\n', ...
    size(tau_avg, 1), size(tau_avg, 2), size(tau_avg, 3));
fprintf('IRF                 | [1, %d]                    | Instrument response\n', ...
    length(IRF));
fprintf('config              | struct                        | Configuration settings\n');
fprintf('-----------------------------------------------------------------\n');

%% Create visualization
fprintf('\nCreating visualization...\n');

% Visualize first image
img_vis = 1;

figure('Position', [100, 100, 1400, 900]);

% Original 512x512
subplot(3, 4, 1);
imagesc(squeeze(Int_512(img_vis, :, :)));
colorbar;
title('Original 512x512 Intensity');
axis image;

% Downsampled 256x256
subplot(3, 4, 2);
imagesc(squeeze(Int_256(img_vis, :, :)));
colorbar;
title('Downsampled 256x256 Intensity');
axis image;

% Lifetime component 1
subplot(3, 4, 3);
imagesc(squeeze(tau_components(img_vis, :, :, 1)));
colorbar;
title('Lifetime Component 1 (ns)');
axis image;

% Lifetime component 2
subplot(3, 4, 4);
imagesc(squeeze(tau_components(img_vis, :, :, 2)));
colorbar;
title('Lifetime Component 2 (ns)');
axis image;

% Fraction component 1
subplot(3, 4, 5);
imagesc(squeeze(f_components(img_vis, :, :, 1)));
colorbar;
title('Fraction Component 1');
axis image;
caxis([0, 1]);

% Fraction component 2
subplot(3, 4, 6);
imagesc(squeeze(f_components(img_vis, :, :, 2)));
colorbar;
title('Fraction Component 2');
axis image;
caxis([0, 1]);

% Average lifetime
subplot(3, 4, 7);
imagesc(squeeze(tau_avg(img_vis, :, :)));
colorbar;
title('Average Lifetime (ns)');
axis image;

% Total photon map
subplot(3, 4, 8);
photon_map = squeeze(sum(Hist(img_vis, :, :, :), 4));
imagesc(photon_map);
colorbar;
title('Total Photons per Pixel');
axis image;

% Sample decay curves
subplot(3, 4, 9);
y_sample = [64, 128, 192];
x_sample = [64, 128, 192];
for i = 1:length(y_sample)
    decay_curve = squeeze(Hist(img_vis, y_sample(i), x_sample(i), :));
    plot(decay_curve, 'LineWidth', 1.5, 'DisplayName', ...
        sprintf('(%d,%d)', y_sample(i), x_sample(i)));
    hold on;
end
xlabel('Time Bin');
ylabel('Photon Count');
title('Sample Decay Curves');
legend('Location', 'best');
grid on;

% Lifetime histogram
subplot(3, 4, 10);
tau_all = tau_components(img_vis, :, :, :);
histogram(tau_all(:), 50);
xlabel('Lifetime (ns)');
ylabel('Count');
title('Lifetime Distribution');
grid on;

% Fraction histogram
subplot(3, 4, 11);
f_all = f_components(img_vis, :, :, :);
histogram(f_all(:), 50);
xlabel('Fraction');
ylabel('Count');
title('Fraction Distribution');
grid on;

% Intensity vs photons
subplot(3, 4, 12);
int_flat = squeeze(Int_256(img_vis, :, :));
int_flat = int_flat(:);
photon_flat = photon_map(:);
scatter(int_flat, photon_flat, 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Normalized Intensity');
ylabel('Total Photons');
title('Intensity vs Photon Count');
grid on;

sgtitle(sprintf('Multi-Exponential Image Data (Image %d/%d, %d components)', ...
    img_vis, config.N_images, config.tau_Num), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
fig_name = strrep(config.output_file, '.mat', '_visualization.png');
saveas(gcf, fig_name);
fprintf('Visualization saved to %s\n', fig_name);

%% Final summary
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Generated %d training images:\n', config.N_images);
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('  - %d lifetime components per pixel\n', config.tau_Num);
fprintf('\nOutput file: %s\n', config.output_file);
fprintf('File size: %.1f MB\n', ...
    (config.N_images * 256 * 256 * 256 * 8 / 1024 / 1024) * 4); % Approx
fprintf('=================================================================\n');
