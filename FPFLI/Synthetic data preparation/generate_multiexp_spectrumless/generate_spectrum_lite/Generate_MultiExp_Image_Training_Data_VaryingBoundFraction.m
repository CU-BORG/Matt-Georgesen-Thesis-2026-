%% Generate Multi-Exponential Training Data (VARYING BOUND FRACTION)
% Spatially-correlated tau with broad fraction variation [0.05-0.95]
% @author: mg

clear; clc;

TAU_1_CENTER = 1.0;
TAU_2_CENTER = 4.0;
STD_LEVELS = 10;
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05];
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50];

config.tau_Num = 2;
config.N_images = 400;
config.image_size_input = 512;
config.image_size_output = 256;
config.bin_width = 0.039;
config.bin_Num = 256;

config.image_source_path = 'C:\Users\mcg11923\Thesis\train';
config.image_pattern = '*.png';

config.tau_center = [TAU_1_CENTER, TAU_2_CENTER];

config.photon_scale_min = 500;
config.photon_scale_max = 5000;
config.intensity_threshold = 0.05;

config.spatial_correlation_length = 10;
config.correlation_method = 'intensity-guided';
config.intensity_modulation = 0.3;

config.f_range = [0.05, 0.95];
config.f_correlation_length = 15;
config.f_intensity_modulation = 0.15;

fprintf('=================================================================\n');
fprintf('Multi-Exponential Image Training Data Generation\n');
fprintf('(SPATIAL CORR + VARYING BOUND FRACTION)\n');
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
fprintf('\nSpatial Correlation Settings (tau maps):\n');
fprintf('  Method: %s\n', config.correlation_method);
fprintf('  Tau correlation length: %d pixels\n', config.spatial_correlation_length);
fprintf('  Tau intensity modulation: %.2f\n', config.intensity_modulation);
fprintf('\nVarying Bound Fraction Settings:\n');
fprintf('  Fraction range: [%.2f, %.2f]\n', config.f_range(1), config.f_range(2));
fprintf('  Fraction correlation length: %d pixels\n', config.f_correlation_length);
fprintf('  Fraction intensity modulation: %.2f\n', config.f_intensity_modulation);
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
    config.output_dir = sprintf('varyingbf_level_%02d_std_%.2f_%.2f', ...
        std_level, TAU_1_STD, TAU_2_STD);

    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end

    fprintf('-----------------------------------------------------------------\n');
    fprintf('LEVEL %d/%d: Tau1_std=%.2f ns, Tau2_std=%.2f ns (VARYING BF)\n', ...
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
    img_normalized = double(img_gray) / 255.0;

    % Downsample to 256x256 using bilinear interpolation
    img_256 = imresize(img_normalized, [config.image_size_output, config.image_size_output], 'bilinear');
    Int = img_256;

    %% Normalize intensity for modulation (used by both tau and fraction maps)
    intensity_normalized = (img_256 - min(img_256(:))) / (max(img_256(:)) - min(img_256(:)) + 1e-10);

    %% Generate SPATIALLY-CORRELATED tau maps (same logic as SpatialCorr)
    tau_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    for comp = 1:config.tau_Num
        % Generate random noise field
        noise_field = randn(config.image_size_output, config.image_size_output);

        % Apply Gaussian smoothing to create spatial correlation
        smooth_noise = imgaussfilt(noise_field, config.spatial_correlation_length);

        % Normalize smooth noise to have std=1
        smooth_noise = smooth_noise / (std(smooth_noise(:)) + 1e-10);

        if strcmp(config.correlation_method, 'intensity-guided')
            % Intensity-guided: modulate tau by intensity structure
            intensity_effect = config.intensity_modulation * (intensity_normalized - 0.5) * 2;
            tau_map(:, :, comp) = config.tau_center(comp) + ...
                config.tau_std(comp) * (smooth_noise + intensity_effect);
        else
            % Pure Gaussian spatial correlation
            tau_map(:, :, comp) = config.tau_center(comp) + ...
                config.tau_std(comp) * smooth_noise;
        end

        % Clamp to reasonable bounds (positive and not too extreme)
        tau_min = max(0.1, config.tau_center(comp) - 3*config.tau_std(comp));
        tau_max = config.tau_center(comp) + 3*config.tau_std(comp);
        tau_map(:, :, comp) = max(tau_min, min(tau_max, tau_map(:, :, comp)));
    end

    %% Generate spatially-varying fraction maps
    f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    noise_f = randn(config.image_size_output, config.image_size_output);
    smooth_noise_f = imgaussfilt(noise_f, config.f_correlation_length);

    if config.f_intensity_modulation ~= 0
        intensity_bias = config.f_intensity_modulation * (intensity_normalized - 0.5) * 2;
        smooth_noise_f = smooth_noise_f + intensity_bias;
    end

    f_min_val = min(smooth_noise_f(:));
    f_max_val = max(smooth_noise_f(:));
    f_normalized = (smooth_noise_f - f_min_val) / (f_max_val - f_min_val + 1e-10);

    F_MIN = config.f_range(1);
    F_MAX = config.f_range(2);
    f_map(:, :, 1) = F_MIN + f_normalized * (F_MAX - F_MIN);
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
            tau_pixel = squeeze(tau_map(y, x, :))';  % Spatially-correlated
            f_pixel = squeeze(f_map(y, x, :))';      % Broadly-varying

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
    output_filename = sprintf('Sample_%03d_varyingbf.mat', img_idx);
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
first_file = fullfile(config.output_dir, 'Sample_001_varyingbf.mat');
sample_data = load(first_file);

fprintf('\nData Statistics (based on Sample_001):\n');
fprintf('-----------------------------------------------------------------\n');

fprintf('Image Sizes:\n');
fprintf('  Input: %dx%d\n', config.image_size_input, config.image_size_input);
fprintf('  Output: %dx%d\n', config.image_size_output, config.image_size_output);
fprintf('  Time bins: %d\n', config.bin_Num);

fprintf('\nLifetime Components (SPATIALLY-CORRELATED):\n');
for comp = 1:config.tau_Num
    tau_data = sample_data.tau_gt_components(:, :, comp);
    tau_data_nonzero = tau_data(tau_data > 1e-6);
    fprintf('  Component %d:\n', comp);
    fprintf('    Target: Gaussian(mu=%.2f ns, sigma=%.3f ns)\n', ...
        config.tau_center(comp), config.tau_std(comp));
    fprintf('    Observed range: [%.3f, %.3f] ns\n', min(tau_data_nonzero), max(tau_data_nonzero));
    fprintf('    Observed mean: %.3f ns, Std: %.3f ns\n', mean(tau_data_nonzero), std(tau_data_nonzero));
end

fprintf('\nFraction Components (BROADLY VARYING):\n');
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
fprintf('tau_gt_components   | [%d, %d, %d]     | Spatially-correlated lifetimes\n', ...
    size(sample_data.tau_gt_components, 1), size(sample_data.tau_gt_components, 2), ...
    size(sample_data.tau_gt_components, 3));
fprintf('f_gt_components     | [%d, %d, %d]     | Broadly-varying fractions\n', ...
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

% Lifetime component 1 (Spatially correlated)
subplot(3, 4, 2);
imagesc(sample_data.tau_gt_components(:, :, 1));
colorbar;
title(sprintf('Tau1 (Spatial Corr: mu=%.2f, sigma=%.3f ns)', TAU_1_CENTER, TAU_1_STD));
axis image;

% Lifetime component 2 (Spatially correlated)
subplot(3, 4, 3);
imagesc(sample_data.tau_gt_components(:, :, 2));
colorbar;
title(sprintf('Tau2 (Spatial Corr: mu=%.2f, sigma=%.3f ns)', TAU_2_CENTER, TAU_2_STD));
axis image;

% Average lifetime
subplot(3, 4, 4);
imagesc(sample_data.tau_gt_avg);
colorbar;
title('Average Lifetime (ns)');
axis image;

% Fraction component 1 — the key plot: should show broad spatial variation
subplot(3, 4, 5);
imagesc(sample_data.f_gt_components(:, :, 1));
colorbar;
title(sprintf('Bound Fraction (Comp 1) [%.2f-%.2f]', F_MIN, F_MAX));
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

% Sample decay curves — pick pixels with very different bound fractions
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

% Fraction histogram — should show broad, roughly uniform distribution
subplot(3, 4, 9);
f1_all = sample_data.f_gt_components(:, :, 1);
f1_all = f1_all(f1_all > 1e-6);
histogram(f1_all, 50);
xlabel('Bound Fraction (Component 1)');
ylabel('Count');
title('Bound Fraction Distribution');
grid on;

% Tau histogram
subplot(3, 4, 10);
tau_all = sample_data.tau_gt_components(:);
tau_all = tau_all(tau_all > 1e-6);
histogram(tau_all, 50);
xlabel('Lifetime (ns)');
ylabel('Count');
title('Lifetime Distribution (Spatial Corr)');
grid on;

% Intensity vs bound fraction — shows intensity-guided modulation effect
subplot(3, 4, 11);
int_flat = sample_data.Int(:);
f1_flat = f1_map(:);
mask = int_flat > 0.05;
scatter(int_flat(mask), f1_flat(mask), 5, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Normalized Intensity');
ylabel('Bound Fraction');
title('Intensity vs Bound Fraction');
grid on;

% Decay curve comparison across samples
subplot(3, 4, 12);
sample_indices = [1, 50, 100];
for i = 1:length(sample_indices)
    sample_file = fullfile(config.output_dir, sprintf('Sample_%03d_varyingbf.mat', sample_indices(i)));
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

sgtitle(sprintf('Level %d VARYING BF (tau1=%.2f+/-%.3f, tau2=%.2f+/-%.3f ns, f1=[%.2f,%.2f])', ...
    std_level, TAU_1_CENTER, TAU_1_STD, TAU_2_CENTER, TAU_2_STD, F_MIN, F_MAX), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fig_name = fullfile(config.output_dir, sprintf('varyingbf_level_%02d_visualization.png', std_level));
try
    saveas(gcf, fig_name);
    fprintf('  Visualization saved to %s\n\n', fig_name);
catch ME
    fprintf('  Warning: Could not save visualization: %s\n\n', ME.message);
end
close(gcf);

end  % End of std_level loop

%% Final summary
total_time = toc(total_tic);
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (VARYING BOUND FRACTION)!\n');
fprintf('=================================================================\n');
fprintf('Generated %d standard deviation levels with VARYING BOUND FRACTION:\n', STD_LEVELS);
fprintf('  - Tau centers: %.2f ns, %.2f ns\n', TAU_1_CENTER, TAU_2_CENTER);
fprintf('  - Std dev range: [%.2f-%.2f ns] for tau1, [%.2f-%.2f ns] for tau2\n', ...
    TAU_1_STD_ARRAY(1), TAU_1_STD_ARRAY(end), TAU_2_STD_ARRAY(1), TAU_2_STD_ARRAY(end));
fprintf('  - Tau spatial correlation length: %d pixels\n', config.spatial_correlation_length);
fprintf('  - Tau correlation method: %s\n', config.correlation_method);
fprintf('  - Fraction range: [%.2f, %.2f] (broad variation)\n', config.f_range(1), config.f_range(2));
fprintf('  - Fraction correlation length: %d pixels\n', config.f_correlation_length);
fprintf('  - Fraction intensity modulation: %.2f\n', config.f_intensity_modulation);
fprintf('  - Images per level: %d\n', config.N_images);
fprintf('  - Total images: %d\n', STD_LEVELS * config.N_images);
fprintf('  - Input: %dx%d intensity images\n', config.image_size_input, config.image_size_input);
fprintf('  - Output: %dx%dx%d data cubes per file\n', ...
    config.image_size_output, config.image_size_output, config.bin_Num);
fprintf('\nOutput directories created:\n');
for i = 1:STD_LEVELS
    dir_name = sprintf('  varyingbf_level_%02d_std_%.2f_%.2f/', i, TAU_1_STD_ARRAY(i), TAU_2_STD_ARRAY(i));
    fprintf('%s\n', dir_name);
end
fprintf('\nFiles per directory: Sample_001_varyingbf.mat through Sample_%03d_varyingbf.mat\n', config.N_images);
fprintf('Approx file size per image: %.1f MB\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5);
fprintf('Total size: ~%.1f MB (~%.1f GB)\n', ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * STD_LEVELS * config.N_images, ...
    (256 * 256 * 256 * 8 / 1024 / 1024) * 1.5 * STD_LEVELS * config.N_images / 1024);
fprintf('Total runtime: %.1f seconds (%.2f minutes, %.2f hours)\n', ...
    total_time, total_time/60, total_time/3600);
