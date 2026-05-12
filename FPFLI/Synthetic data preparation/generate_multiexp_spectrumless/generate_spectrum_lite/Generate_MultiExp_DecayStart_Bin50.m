%% Generate Multi-Exponential Training Data (DECAY PEAK AT BIN 62)
% ALL parameters (tau1, tau2, f1) drawn from UNIFORM distributions
% NO spatial correlation for any parameter
% Decay shifted to BIN 62 via IRF CONVOLUTION (matched to REAL DATA)
%
% METHOD: Uses IRF centered at bin 62, which when convolved with the
% exponential decay naturally produces a decay peaking at bin 62.
% This matches the real FLIM data temporal alignment.
%
% @author: mg

clear; clc;

%% Tau ranges - similar to real data statistics

TAU_1_MIN = 0.20;   % Narrower range for better learning
TAU_1_MAX = 0.70;
TAU_2_MIN = 1.20;   % Wider range than real data
TAU_2_MAX = 4.50;

%% IRF timing parameter (controls decay shift via convolution)
IRF_CENTER_BIN = 62;  % IRF centered at bin 62 -> decay peaks near this bin
                      % Matched to real data peak location (bin 62)
                      % Convolution with exp(-t/tau) naturally shifts decay

%% Configuration
config.tau_Num = 2;              % Always 2 components
config.N_images = 100;           % Number of training images (100 for this experiment)
config.image_size_input = 512;   % Input image size
config.image_size_output = 256;  % Output image size (downsampled)
config.bin_width = 0.048828;     % Time bin width in nanoseconds (12.5ns / 256 bins)
config.bin_Num = 256;            % Number of time bins

% Path to 512x512 intensity images (PNG format)
config.image_source_path = 'C:\Users\mcg11923\Thesis\train';
config.image_pattern = '*.png';

% Tau ranges (uniform random)
config.tau_ranges = [TAU_1_MIN, TAU_1_MAX; TAU_2_MIN, TAU_2_MAX];

% Photon count - FLAT RANDOM (not intensity-based)
config.photon_scale_min = 50;     % Minimum photons per pixel
config.photon_scale_max = 1000;   % Maximum photons per pixel
config.intensity_threshold = 0.05; % Pixels below this intensity get no photons

% Fraction range (uniform random)
config.f_range = [0.1, 0.9];

fprintf('=================================================================\n');
fprintf('Multi-Exponential Image Training Data Generation\n');
fprintf('(DECAY PEAK BIN 62 - MATCHED TO REAL DATA TEMPORAL ALIGNMENT)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  IRF center bin: %d (decay timing via convolution)\n', IRF_CENTER_BIN);
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
fprintf('  Tau1 (SHORT): Uniform[%.2f, %.2f] ns (per pixel, independent)\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  Tau2 (LONG):  Uniform[%.2f, %.2f] ns (per pixel, independent)\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  Bound Fraction f (tau2): Uniform[%.2f, %.2f] (per pixel, independent)\n', config.f_range(1), config.f_range(2));
fprintf('  Free Fraction (1-f) (tau1): Uniform[%.2f, %.2f] (per pixel, independent)\n', 1-config.f_range(2), 1-config.f_range(1));
fprintf('  NO SPATIAL CORRELATION for any parameter\n');
fprintf('-----------------------------------------------------------------\n');

%% Generate IRF centered at specified bin (controls decay timing via convolution)
fprintf('\nGenerating Gaussian IRF for decay shift...\n');
fprintf('  IRF center: bin %d (t = %.3f ns)\n', IRF_CENTER_BIN, IRF_CENTER_BIN * config.bin_width);
fprintf('  IRF width (FWHM): 0.120 ns = 120 ps\n');

% Generate Gaussian IRF centered at IRF_CENTER_BIN
IRF = IRF_gaussian(IRF_CENTER_BIN, config.bin_width, 0.120);

% Normalize IRF to unit amplitude
IRF = IRF / max(IRF);

fprintf('  IRF generated successfully\n');
fprintf('  --> When convolved with exp(-t/tau), produces decay peaking near bin %d\n', IRF_CENTER_BIN);

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
config.output_dir = 'E:\decay_peak_bin62_12.5ns_trainfixed';

if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('\n=================================================================\n');
fprintf('Starting generation of %d images...\n', config.N_images);
fprintf('  Decay timing: IRF convolution shifts peak to bin %d\n', IRF_CENTER_BIN);
fprintf('  Output directory: %s\n', config.output_dir);
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
    % f represents the fraction of photons from tau2 (LONG lifetime)
    f_map = zeros(config.image_size_output, config.image_size_output, config.tau_Num);

    F_MIN = config.f_range(1);
    F_MAX = config.f_range(2);

    % f (Component 2): Fraction of LONG lifetime (tau2) - Uniform random [F_MIN, F_MAX]
    f_map(:, :, 2) = F_MIN + (F_MAX - F_MIN) * rand(config.image_size_output, config.image_size_output);
    % (1-f) (Component 1): Fraction of SHORT lifetime (tau1)
    f_map(:, :, 1) = 1 - f_map(:, :, 2);

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

            % Generate FLAT RANDOM photon count (independent of intensity)
            N_photons = randi([config.photon_scale_min, config.photon_scale_max]);

            % Get lifetime components and fractions for this pixel
            tau_pixel = squeeze(tau_map(y, x, :))';  % Flat random
            f_pixel = squeeze(f_map(y, x, :))';      % Flat random

            % Generate multi-exponential decay via IRF convolution
            % IRF convolution shifts decay to center at IRF_CENTER_BIN
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
    output_filename = sprintf('Sample_%03d_bin62_12.5ns.mat', img_idx);
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
first_file = fullfile(config.output_dir, 'Sample_001_bin62_12.5ns.mat');
sample_data = load(first_file);

fprintf('\n=================================================================\n');
fprintf('Data Statistics (based on Sample_001):\n');
fprintf('=================================================================\n');

fprintf('\nTiming:\n');
fprintf('  IRF center: Bin %d (t = %.3f ns)\n', IRF_CENTER_BIN, IRF_CENTER_BIN * config.bin_width);
fprintf('  Decay peak location: Near bin %d (via IRF convolution)\n', IRF_CENTER_BIN);

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

fprintf('\nPhoton Counts (Sample_001):\n');
total_photons = sum(sample_data.Hist(:));
fprintf('  Total photons: %.0f\n', total_photons);

%% Create visualization
fprintf('\nCreating visualization...\n');

figure('Position', [100, 100, 1400, 900]);

% Intensity
subplot(3, 4, 1);
imagesc(sample_data.Int);
colorbar;
title('256x256 Intensity');
axis image;

% Lifetime component 1 (SHORT - Flat random)
subplot(3, 4, 2);
imagesc(sample_data.tau_gt_components(:, :, 1));
colorbar;
title(sprintf('Tau1 SHORT (Uniform [%.2f, %.2f] ns)', TAU_1_MIN, TAU_1_MAX));
axis image;

% Lifetime component 2 (LONG - Flat random)
subplot(3, 4, 3);
imagesc(sample_data.tau_gt_components(:, :, 2));
colorbar;
title(sprintf('Tau2 LONG (Uniform [%.2f, %.2f] ns)', TAU_2_MIN, TAU_2_MAX));
axis image;

% Average lifetime
subplot(3, 4, 4);
imagesc(sample_data.tau_gt_avg);
colorbar;
title('Average Lifetime (ns)');
axis image;

% Fraction component 1 (FREE fraction for SHORT tau1)
subplot(3, 4, 5);
imagesc(sample_data.f_gt_components(:, :, 1));
colorbar;
title(sprintf('Free Fraction (1-f) [%.2f-%.2f]', 1-F_MAX, 1-F_MIN));
axis image;
caxis([0, 1]);

% Fraction component 2 (BOUND fraction for LONG tau2)
subplot(3, 4, 6);
imagesc(sample_data.f_gt_components(:, :, 2));
colorbar;
title(sprintf('Bound Fraction (f) [%.2f-%.2f]', F_MIN, F_MAX));
axis image;
caxis([0, 1]);

% Total photon map
subplot(3, 4, 7);
photon_map = sum(sample_data.Hist, 3);
imagesc(photon_map);
colorbar;
title('Total Photons per Pixel');
axis image;

% Sample decay curve showing bin 50 start
subplot(3, 4, 8);
decay_curve = squeeze(sample_data.Hist(128, 128, :));
plot(decay_curve, 'LineWidth', 2);
hold on;
xline(IRF_CENTER_BIN, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('IRF center (bin %d)', IRF_CENTER_BIN));
xlabel('Time Bin');
ylabel('Photon Count');
title(sprintf('Decay Curve (center pixel) - Peak Bin %d', IRF_CENTER_BIN));
legend('Location', 'best');
grid on;

% Fraction histogram (BOUND fraction = Component 2 = LONG tau2)
subplot(3, 4, 9);
f_bound = sample_data.f_gt_components(:, :, 2);
f_bound = f_bound(f_bound > 1e-6);
histogram(f_bound, 50);
xlabel('Bound Fraction f (LONG tau2)');
ylabel('Count');
title('Bound Fraction Distribution');
grid on;

% Tau1 histogram (SHORT lifetime)
subplot(3, 4, 10);
tau1_all = sample_data.tau_gt_components(:, :, 1);
tau1_all = tau1_all(tau1_all > 1e-6);
histogram(tau1_all, 50);
xlabel('Tau1 SHORT (ns)');
ylabel('Count');
title('Tau1 Distribution (SHORT)');
grid on;

% Tau2 histogram (LONG lifetime)
subplot(3, 4, 11);
tau2_all = sample_data.tau_gt_components(:, :, 2);
tau2_all = tau2_all(tau2_all > 1e-6);
histogram(tau2_all, 50);
xlabel('Tau2 LONG (ns)');
ylabel('Count');
title('Tau2 Distribution (LONG)');
grid on;

% Temporal profile - sum over all pixels
subplot(3, 4, 12);
temporal_profile = squeeze(sum(sum(sample_data.Hist, 1), 2));
plot(temporal_profile, 'LineWidth', 2);
hold on;
xline(IRF_CENTER_BIN, '--r', 'LineWidth', 1.5);
xlabel('Time Bin');
ylabel('Total Photon Count');
title(sprintf('Temporal Profile (all pixels, IRF bin %d)', IRF_CENTER_BIN));
grid on;

sgtitle(sprintf('DECAY PEAK BIN %d via IRF Convolution (tau1=[%.2f,%.2f], tau2=[%.2f,%.2f] ns, f1=[%.2f,%.2f])', ...
    IRF_CENTER_BIN, TAU_1_MIN, TAU_1_MAX, TAU_2_MIN, TAU_2_MAX, F_MIN, F_MAX), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
fig_name = fullfile(config.output_dir, sprintf('decay_bin%d_visualization.png', IRF_CENTER_BIN));
try
    saveas(gcf, fig_name);
    fprintf('  Visualization saved to %s\n\n', fig_name);
catch ME
    fprintf('  Warning: Could not save visualization: %s\n\n', ME.message);
end
close(gcf);

%% Final summary

fprintf('DATA GENERATION COMPLETE (DECAY PEAK BIN %d via IRF)!\n', IRF_CENTER_BIN);
fprintf('=================================================================\n');
fprintf('Generated %d images for temporal alignment experiment:\n', config.N_images);
fprintf('  - IRF center: Bin %d (t = %.3f ns)\n', IRF_CENTER_BIN, IRF_CENTER_BIN * config.bin_width);
fprintf('  - Decay timing: Shifted via IRF convolution\n');
fprintf('  - Tau1 (SHORT): Uniform[%.2f, %.2f] ns\n', TAU_1_MIN, TAU_1_MAX);
fprintf('  - Tau2 (LONG):  Uniform[%.2f, %.2f] ns\n', TAU_2_MIN, TAU_2_MAX);
fprintf('  - Bound Fraction f (tau2): Uniform[%.2f, %.2f]\n', F_MIN, F_MAX);
fprintf('  - Free Fraction (1-f) (tau1): Uniform[%.2f, %.2f]\n', 1-F_MAX, 1-F_MIN);
fprintf('  - NO spatial correlation for any parameter\n');
fprintf('\nOutput directory: %s/\n', config.output_dir);
fprintf('Files: Sample_001_bin62_12.5ns.mat through Sample_%03d_bin62_12.5ns.mat\n', config.N_images);
fprintf('Total runtime: %.1f seconds (%.2f minutes)\n', total_time, total_time/60);


