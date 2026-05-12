% visualize_sample_decays.m
% Visualizes sample bi-exponential fluorescence decays from synthetic training data
% Shows multiple examples with different photon counts and lifetime parameters

clear; close all; clc;

%% Configuration
DATA_DIR = 'E:\decay_peak_bin62_12.5ns_train';
N_SAMPLES = 4;  % Number of sample decays to show
BIN_WIDTH = 0.048828;  % ns per bin
N_BINS = 256;

%% Create time axis in nanoseconds
time_ns = (0:N_BINS-1) * BIN_WIDTH;

%% Load random samples from training data
mat_files = dir(fullfile(DATA_DIR, '*.mat'));
if isempty(mat_files)
    error('No .mat files found in %s', DATA_DIR);
end

fprintf('Found %d training samples\n', length(mat_files));
fprintf('Loading %d random samples for visualization...\n\n', N_SAMPLES);

% Select random files
rng(42);  % For reproducibility
random_indices = randperm(length(mat_files), N_SAMPLES);

%% Create figure with subplots
fig = figure('Position', [100, 100, 1400, 900]);

for idx = 1:N_SAMPLES
    file_idx = random_indices(idx);
    file_path = fullfile(mat_files(file_idx).folder, mat_files(file_idx).name);

    % Load data
    data = load(file_path);

    % Extract single pixel (use middle of image)
    [h, w, ~] = size(data.Hist);
    pixel_row = round(h/2);
    pixel_col = round(w/2);

    % Get decay curve and parameters
    decay = squeeze(data.Hist(pixel_row, pixel_col, :));
    tau1 = data.tau_gt_components(pixel_row, pixel_col, 1);
    tau2 = data.tau_gt_components(pixel_row, pixel_col, 2);
    f1 = data.f_gt_components(pixel_row, pixel_col, 1);
    f2 = data.f_gt_components(pixel_row, pixel_col, 2);
    photons = sum(decay);

    % Create subplot
    subplot(2, 2, idx);

    % Plot decay on semi-log scale
    semilogy(time_ns, decay, 'b-', 'LineWidth', 1.5);
    hold on;

    % Mark the IRF center (bin 62)
    irf_time = 62 * BIN_WIDTH;
    ylim_vals = ylim;
    plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 1.5);

    % Formatting
    grid on;
    xlabel('Time (ns)', 'FontSize', 11);
    ylabel('Photon Counts (log scale)', 'FontSize', 11);
    title(sprintf('Sample %d: \\tau_1=%.2f ns, \\tau_2=%.2f ns, f=%.2f, N=%d photons', ...
        idx, tau1, tau2, f2, round(photons)), 'FontSize', 10, 'FontWeight', 'bold');
    xlim([0, 12.5]);

    % Add legend
    if idx == 1
        legend({'Decay curve', 'IRF center (3.03 ns)'}, 'Location', 'northeast', 'FontSize', 9);
    end

    % Print info
    fprintf('Sample %d:\n', idx);
    fprintf('  File: %s\n', mat_files(file_idx).name);
    fprintf('  tau1 = %.3f ns\n', tau1);
    fprintf('  tau2 = %.3f ns\n', tau2);
    fprintf('  f (slow fraction) = %.3f\n', f2);
    fprintf('  Photons = %d\n', round(photons));
    fprintf('  Peak bin: %d (%.3f ns)\n', find(decay == max(decay), 1), ...
        find(decay == max(decay), 1) * BIN_WIDTH);
    fprintf('\n');
end

% Add overall title
sgtitle('Sample Bi-Exponential Fluorescence Decays from Training Data', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
output_path = 'C:\Users\mcg11923\Thesis\sample_decays_visualization.png';
saveas(fig, output_path);
fprintf('Figure saved to: %s\n', output_path);

%% Additional figure: Show effect of varying photon counts with same lifetimes
fprintf('\n=== Creating photon count comparison figure ===\n');

% Find samples with similar lifetimes but different photon counts
all_data = struct('tau1', [], 'tau2', [], 'f', [], 'photons', [], 'decay', [], 'filename', {});

for i = 1:min(50, length(mat_files))  % Sample first 50 files
    file_path = fullfile(mat_files(i).folder, mat_files(i).name);
    data = load(file_path);

    % Extract center pixel
    [h, w, ~] = size(data.Hist);
    pixel_row = round(h/2);
    pixel_col = round(w/2);

    decay = squeeze(data.Hist(pixel_row, pixel_col, :));
    all_data(i).tau1 = data.tau_gt_components(pixel_row, pixel_col, 1);
    all_data(i).tau2 = data.tau_gt_components(pixel_row, pixel_col, 2);
    all_data(i).f = data.f_gt_components(pixel_row, pixel_col, 2);
    all_data(i).photons = sum(decay);
    all_data(i).decay = decay;
    all_data(i).filename = mat_files(i).name;
end

% Find samples with tau1 ~ 0.5 ns, tau2 ~ 3.0 ns, f ~ 0.7
% but varying photon counts
target_tau1 = 0.5;
target_tau2 = 3.0;
target_f = 0.7;

tau1_match = abs([all_data.tau1] - target_tau1) < 0.1;
tau2_match = abs([all_data.tau2] - target_tau2) < 0.5;
f_match = abs([all_data.f] - target_f) < 0.15;

matching_indices = find(tau1_match & tau2_match & f_match);

if length(matching_indices) >= 3
    % Sort by photon count
    photon_counts = [all_data(matching_indices).photons];
    [~, sort_idx] = sort(photon_counts);

    % Select low, medium, high photon counts
    selected = matching_indices(sort_idx([1, round(end/2), end]));

    fig2 = figure('Position', [150, 150, 1200, 400]);

    for plot_idx = 1:3
        idx = selected(plot_idx);

        subplot(1, 3, plot_idx);
        semilogy(time_ns, all_data(idx).decay, 'b-', 'LineWidth', 1.5);
        hold on;

        % Mark IRF center
        irf_time = 62 * BIN_WIDTH;
        ylim_vals = ylim;
        plot([irf_time, irf_time], ylim_vals, 'r--', 'LineWidth', 1.5);

        grid on;
        xlabel('Time (ns)', 'FontSize', 11);
        ylabel('Photon Counts (log scale)', 'FontSize', 11);
        title(sprintf('N=%d photons\n\\tau_1=%.2f ns, \\tau_2=%.2f ns, f=%.2f', ...
            round(all_data(idx).photons), all_data(idx).tau1, all_data(idx).tau2, ...
            all_data(idx).f), 'FontSize', 10, 'FontWeight', 'bold');
        xlim([0, 12.5]);

        fprintf('Photon comparison sample %d:\n', plot_idx);
        fprintf('  tau1 = %.3f ns\n', all_data(idx).tau1);
        fprintf('  tau2 = %.3f ns\n', all_data(idx).tau2);
        fprintf('  f = %.3f\n', all_data(idx).f);
        fprintf('  Photons = %d\n', round(all_data(idx).photons));
        fprintf('\n');
    end

    sgtitle('Effect of Photon Count on Decay Measurement (Similar Lifetimes)', ...
        'FontSize', 14, 'FontWeight', 'bold');

    output_path2 = 'C:\Users\mcg11923\Thesis\photon_count_comparison.png';
    saveas(fig2, output_path2);
    fprintf('Photon comparison figure saved to: %s\n', output_path2);
else
    fprintf('Not enough matching samples found for photon count comparison\n');
end

fprintf('\nVisualization complete!\n');
