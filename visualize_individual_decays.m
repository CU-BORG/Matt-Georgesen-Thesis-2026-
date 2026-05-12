% visualize_individual_decays.m
% Creates individual decay plots with specific photon counts (~100 and ~900)

clear; close all; clc;

%% Configuration
DATA_DIR = 'E:\decay_peak_bin62_12.5ns_train';
BIN_WIDTH = 0.048828;  % ns per bin
N_BINS = 256;

%% Create time axis in nanoseconds
time_ns = (0:N_BINS-1) * BIN_WIDTH;

%% Create output directory
individual_dir = 'C:\Users\mcg11923\Thesis\individual_decay_plots';
if ~exist(individual_dir, 'dir')
    mkdir(individual_dir);
end

%% Load sample files
mat_files = dir(fullfile(DATA_DIR, 'Sample_*.mat'));
fprintf('Found %d training samples\n', length(mat_files));

%% Find samples with ~100 and ~900 photons
fprintf('\nSearching for specific photon count samples...\n');

target_photons = [100, 300, 600, 900];
selected_data = cell(4, 1);

for idx = 1:4
    target = target_photons(idx);
    found = false;

    % Search through files (limit to first 50 for speed)
    for file_idx = 1:min(50, length(mat_files))
        file_path = fullfile(mat_files(file_idx).folder, mat_files(file_idx).name);
        data = load(file_path);

        % Find pixels with photon count near target
        photon_counts = squeeze(sum(data.Hist, 3));
        diff = abs(photon_counts - target);
        [min_diff, min_idx] = min(diff(:));

        if min_diff < target * 0.15  % Within 15% of target
            [pixel_row, pixel_col] = ind2sub(size(photon_counts), min_idx);

            selected_data{idx}.file_idx = file_idx;
            selected_data{idx}.filename = mat_files(file_idx).name;
            selected_data{idx}.pixel_row = pixel_row;
            selected_data{idx}.pixel_col = pixel_col;
            selected_data{idx}.decay = squeeze(data.Hist(pixel_row, pixel_col, :));
            selected_data{idx}.tau1 = data.tau_gt_components(pixel_row, pixel_col, 1);
            selected_data{idx}.tau2 = data.tau_gt_components(pixel_row, pixel_col, 2);
            selected_data{idx}.f2 = data.f_gt_components(pixel_row, pixel_col, 2);
            selected_data{idx}.photons = sum(selected_data{idx}.decay);

            fprintf('  Found ~%d photons: File %s, Pixel [%d,%d], Actual=%d photons\n', ...
                target, mat_files(file_idx).name, pixel_row, pixel_col, round(selected_data{idx}.photons));
            found = true;
            break;
        end
    end

    if ~found
        fprintf('  WARNING: Could not find sample for ~%d photons\n', target);
    end
end

%% Create individual plots (log scale)
fprintf('\n=== Creating individual log-scale plots ===\n');

for idx = 1:4
    if isempty(selected_data{idx})
        continue;
    end

    d = selected_data{idx};

    % Create figure
    fig = figure('Position', [100, 100, 800, 600]);

    % Plot on log scale
    semilogy(time_ns, d.decay, 'b-', 'LineWidth', 2.5);
    hold on;

    % Mark IRF center
    irf_time = 62 * BIN_WIDTH;
    ylim_vals = ylim;
    plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 2.5);

    % Formatting
    grid on;
    xlabel('Time (ns)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Photon Counts (log scale)', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('\\tau_1=%.2f ns, \\tau_2=%.2f ns, f_{slow}=%.2f, N=%d photons', ...
        d.tau1, d.tau2, d.f2, round(d.photons)), 'FontSize', 13, 'FontWeight', 'bold');
    xlim([0, 12.5]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    legend({'Decay curve', 'IRF center (3.03 ns)'}, 'Location', 'northeast', 'FontSize', 11);

    % Save
    output_path = fullfile(individual_dir, sprintf('decay_%dphotons_log.png', round(d.photons)));
    saveas(fig, output_path);
    fprintf('  Saved: %s\n', output_path);
    close(fig);
end

%% Create individual plots (linear scale)
fprintf('\n=== Creating individual linear-scale plots ===\n');

for idx = 1:4
    if isempty(selected_data{idx})
        continue;
    end

    d = selected_data{idx};

    % Create figure
    fig = figure('Position', [100, 100, 800, 600]);

    % Plot on linear scale
    plot(time_ns, d.decay, 'b-', 'LineWidth', 2.5);
    hold on;

    % Mark IRF center
    irf_time = 62 * BIN_WIDTH;
    ylim_vals = ylim;
    plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 2.5);

    % Formatting
    grid on;
    xlabel('Time (ns)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Photon Counts', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('\\tau_1=%.2f ns, \\tau_2=%.2f ns, f_{slow}=%.2f, N=%d photons', ...
        d.tau1, d.tau2, d.f2, round(d.photons)), 'FontSize', 13, 'FontWeight', 'bold');
    xlim([0, 12.5]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    legend({'Decay curve', 'IRF center (3.03 ns)'}, 'Location', 'northeast', 'FontSize', 11);

    % Save
    output_path = fullfile(individual_dir, sprintf('decay_%dphotons_linear.png', round(d.photons)));
    saveas(fig, output_path);
    fprintf('  Saved: %s\n', output_path);
    close(fig);
end

fprintf('\n=== Summary ===\n');
for idx = 1:4
    if ~isempty(selected_data{idx})
        d = selected_data{idx};
        fprintf('Sample %d:\n', idx);
        fprintf('  Photons: %d\n', round(d.photons));
        fprintf('  tau1 = %.3f ns, tau2 = %.3f ns, f = %.3f\n', d.tau1, d.tau2, d.f2);
        fprintf('  File: %s [%d, %d]\n\n', d.filename, d.pixel_row, d.pixel_col);
    end
end

fprintf('All plots saved to: %s\n', individual_dir);
fprintf('\nVisualization complete!\n');
