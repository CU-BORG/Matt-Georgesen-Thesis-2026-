% visualize_decay_examples.m
% Creates publication-quality figures showing sample fluorescence decays

clear; close all; clc;

%% Configuration
DATA_DIR = 'E:\decay_peak_bin62_12.5ns_train';
BIN_WIDTH = 0.048828;  % ns per bin
N_BINS = 256;

%% Create time axis in nanoseconds
time_ns = (0:N_BINS-1) * BIN_WIDTH;

%% Load a few specific samples
mat_files = dir(fullfile(DATA_DIR, 'Sample_*.mat'));
fprintf('Found %d training samples\n\n', length(mat_files));

% Find samples with specific photon counts
fprintf('Searching for samples with specific photon counts...\n');

% Pre-scan files to find good candidates
target_photons = [100, 300, 600, 900];  % Target photon counts
selected_samples = [];

for target_count = target_photons
    best_match_idx = -1;
    best_match_diff = inf;

    for file_idx = 1:min(80, length(mat_files))
        file_path = fullfile(mat_files(file_idx).folder, mat_files(file_idx).name);
        data = load(file_path);

        % Find pixels with photon counts near target
        photon_counts = squeeze(sum(data.Hist, 3));
        valid_mask = photon_counts > (target_count * 0.8) & photon_counts < (target_count * 1.2);

        if any(valid_mask(:))
            % Get median photon count of valid pixels
            valid_counts = photon_counts(valid_mask);
            median_count = median(valid_counts);
            diff = abs(median_count - target_count);

            if diff < best_match_diff
                best_match_diff = diff;
                best_match_idx = file_idx;
            end
        end
    end

    if best_match_idx > 0
        selected_samples = [selected_samples; best_match_idx];
        fprintf('  Found sample %d for ~%d photons (diff: %.0f)\n', ...
            best_match_idx, target_count, best_match_diff);
    else
        % Fallback to random sample
        selected_samples = [selected_samples; randi(length(mat_files))];
        fprintf('  Using random sample for ~%d photons (no good match found)\n', target_count);
    end
end

sample_indices = selected_samples';

%% Create individual figures for each sample (log scale)
fprintf('=== Creating individual decay plots (log scale) ===\n');

% Create output directory for individual plots
individual_dir = 'C:\Users\mcg11923\Thesis\individual_decay_plots';
if ~exist(individual_dir, 'dir')
    mkdir(individual_dir);
end

for plot_idx = 1:4
    file_idx = sample_indices(plot_idx);
    file_path = fullfile(mat_files(file_idx).folder, mat_files(file_idx).name);

    % Load data
    data = load(file_path);

    % Find a pixel with good photon count (> 200 photons)
    photon_counts = squeeze(sum(data.Hist, 3));
    valid_mask = photon_counts > 200 & photon_counts < 800;

    [rows, cols] = find(valid_mask);
    if isempty(rows)
        % Fallback to center pixel
        [h, w, ~] = size(data.Hist);
        pixel_row = round(h/2);
        pixel_col = round(w/2);
    else
        % Pick first valid pixel
        pixel_row = rows(1);
        pixel_col = cols(1);
    end

    % Extract decay and parameters
    decay = squeeze(data.Hist(pixel_row, pixel_col, :));
    tau1 = data.tau_gt_components(pixel_row, pixel_col, 1);
    tau2 = data.tau_gt_components(pixel_row, pixel_col, 2);
    f1 = data.f_gt_components(pixel_row, pixel_col, 1);
    f2 = data.f_gt_components(pixel_row, pixel_col, 2);
    photons = sum(decay);

    % Create individual figure (log scale)
    fig = figure('Position', [100, 100, 800, 600]);

    % Plot decay on semi-log scale
    semilogy(time_ns, decay, 'b-', 'LineWidth', 2.5);
    hold on;

    % Mark the IRF center (bin 62)
    irf_time = 62 * BIN_WIDTH;
    ylim_vals = ylim;
    plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 2.5);

    % Formatting
    grid on;
    xlabel('Time (ns)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Photon Counts (log scale)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('\\tau_1=%.2f ns, \\tau_2=%.2f ns, f_{slow}=%.2f, N=%d photons', ...
        tau1, tau2, f2, round(photons)), 'FontSize', 13, 'FontWeight', 'bold');
    xlim([0, 12.5]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);

    % Add legend
    legend({'Decay curve', 'IRF center (3.03 ns)'}, 'Location', 'northeast', 'FontSize', 11);

    % Save individual plot
    output_path = fullfile(individual_dir, sprintf('decay_sample_%d_log.png', plot_idx));
    saveas(fig, output_path);
    close(fig);

    % Print info
    fprintf('Sample %d (File: %s, Pixel [%d, %d]):\n', plot_idx, ...
        mat_files(file_idx).name, pixel_row, pixel_col);
    fprintf('  tau1 = %.3f ns\n', tau1);
    fprintf('  tau2 = %.3f ns\n', tau2);
    fprintf('  f1 = %.3f, f2 = %.3f\n', f1, f2);
    fprintf('  Photons = %d\n', round(photons));
    fprintf('  Peak bin: %d (%.3f ns)\n', find(decay == max(decay), 1), ...
        find(decay == max(decay), 1) * BIN_WIDTH);
    fprintf('  Saved: %s\n\n', output_path);
end

fprintf('Individual log-scale plots saved to: %s\n\n', individual_dir);

%% Create individual figures (linear scale)
fprintf('\n=== Creating individual decay plots (linear scale) ===\n');

for plot_idx = 1:4
    file_idx = sample_indices(plot_idx);
    file_path = fullfile(mat_files(file_idx).folder, mat_files(file_idx).name);

    data = load(file_path);

    % Find pixel with target photon count
    photon_counts = squeeze(sum(data.Hist, 3));
    target_count = target_photons(plot_idx);
    valid_mask = photon_counts > (target_count * 0.8) & photon_counts < (target_count * 1.2);

    [rows, cols] = find(valid_mask);
    if isempty(rows)
        [h, w, ~] = size(data.Hist);
        pixel_row = round(h/2);
        pixel_col = round(w/2);
    else
        pixel_row = rows(1);
        pixel_col = cols(1);
    end

    decay = squeeze(data.Hist(pixel_row, pixel_col, :));
    tau1 = data.tau_gt_components(pixel_row, pixel_col, 1);
    tau2 = data.tau_gt_components(pixel_row, pixel_col, 2);
    f2 = data.f_gt_components(pixel_row, pixel_col, 2);
    photons = sum(decay);

    % Create individual figure (linear scale)
    fig = figure('Position', [100, 100, 800, 600]);

    % Plot decay on linear scale
    plot(time_ns, decay, 'b-', 'LineWidth', 2.5);
    hold on;

    % Mark IRF center
    irf_time = 62 * BIN_WIDTH;
    ylim_vals = ylim;
    plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 2.5);

    grid on;
    xlabel('Time (ns)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Photon Counts', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('\\tau_1=%.2f ns, \\tau_2=%.2f ns, f_{slow}=%.2f, N=%d photons', ...
        tau1, tau2, f2, round(photons)), 'FontSize', 13, 'FontWeight', 'bold');
    xlim([0, 12.5]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);

    legend({'Decay curve', 'IRF center (3.03 ns)'}, 'Location', 'northeast', 'FontSize', 11);

    % Save individual plot
    output_path = fullfile(individual_dir, sprintf('decay_sample_%d_linear.png', plot_idx));
    saveas(fig, output_path);
    close(fig);

    fprintf('Sample %d (Linear): N=%d photons, Saved: %s\n', plot_idx, round(photons), output_path);
end

fprintf('\nIndividual linear-scale plots saved to: %s\n', individual_dir);

fprintf('\nVisualization complete!\n');
