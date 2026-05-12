%% Generate Multi-Exponential LLE Training Data
% Creates training data for multi-exponential Local Lifetime Estimation
%
% Output: Training_data_multiexp.mat containing:
%   - y: Fluorescence decay histograms [N_samples, 256]
%   - irf: Instrument response functions [N_samples, 256]
%   - tau_components: Individual lifetime values [N_samples, n_components]
%   - f_components: Individual fraction values [N_samples, n_components]
%   - tau_ave: Average lifetimes [N_samples, 1]
%
% Author: MG
% Date: 2025-09-30

clear; clc;

%% Configuration
config.tau_Num = 2;              % Number of lifetime components (2 or 3)
config.N_samples = 10000;        % Number of training samples
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins
config.output_file = 'Training_data_multiexp.mat';

% Lifetime ranges for each component (in nanoseconds)
% Component 1: Short lifetime (typical: 0.5-2.5 ns)
% Component 2: Long lifetime (typical: 2.5-5.0 ns)
% Component 3: Very long lifetime (typical: 5.0-8.0 ns) - if tau_Num=3
config.tau_ranges = [
    0.4, 2.5;   % Component 1 range
    2.5, 5.0;   % Component 2 range
    5.0, 8.0    % Component 3 range (only used if tau_Num=3)
];

% Photon count range (varying to simulate different intensities)
config.N_photon_min = 100;
config.N_photon_max = 500;


fprintf('Multi-Exponential LLE Training Data Generation\n');

fprintf('Configuration:\n');
fprintf('  Number of components: %d\n', config.tau_Num);
fprintf('  Training samples: %d\n', config.N_samples);
fprintf('  Bin width: %.4f ns\n', config.bin_width);
fprintf('  Time bins: %d\n', config.bin_Num);
fprintf('  Photon count range: [%d, %d]\n', config.N_photon_min, config.N_photon_max);
fprintf('  Output file: %s\n', config.output_file);
fprintf('-----------------------------------------------------------------\n');

%% Generate or load IRF
% Option 1: Load existing IRF
irf_file = 'irf_measure.mat';
if exist(irf_file, 'file')
    fprintf('Loading IRF from %s...\n', irf_file);
    irf_data = load(irf_file);
    IRF = irf_data.irf;
else
    % Option 2: Generate Gaussian IRF
    fprintf('Generating Gaussian IRF...\n');
    IRF = IRF_gaussian(14, config.bin_width, 0.1673);
end

% Normalize IRF
IRF = IRF / max(IRF);
fprintf('IRF loaded/generated successfully\n');

%% Initialize output arrays
y = zeros(config.N_samples, config.bin_Num);
tau_components = zeros(config.N_samples, config.tau_Num);
f_components = zeros(config.N_samples, config.tau_Num);
tau_ave = zeros(config.N_samples, 1);

%% Generate training samples
fprintf('\nGenerating %d multi-exponential decay samples...\n', config.N_samples);
fprintf('Progress: ');

tic;
for i = 1:config.N_samples
    %% Generate random lifetime components
    % Each component drawn from its respective range
    tau = zeros(1, config.tau_Num);
    for comp = 1:config.tau_Num
        tau_min = config.tau_ranges(comp, 1);
        tau_max = config.tau_ranges(comp, 2);
        tau(comp) = tau_min + rand() * (tau_max - tau_min);
    end

    %% Generate random fraction components
    % Method: Dirichlet distribution (uniform over simplex)
    % This ensures fractions are positive and sum to 1
    alpha = ones(1, config.tau_Num);  % Uniform distribution
    f = gamrnd(alpha, 1);
    f = f / sum(f);  % Normalize to sum to 1

    % Alternative: Uniform random fractions (uncomment to use)
    % f = rand(1, config.tau_Num);
    % f = f / sum(f);

    %% Generate photon count (varying intensity)
    N_photons = config.N_photon_min + randi(config.N_photon_max - config.N_photon_min);

    %% Generate fluorescence decay histogram
    y(i,:) = Fluorescence_multi_decay_nonhomopp(...
        config.tau_Num, tau, f, N_photons, config.bin_width, IRF);

    %% Store ground truth components
    tau_components(i,:) = tau;
    f_components(i,:) = f;
    tau_ave(i) = sum(tau .* f);  % Calculate average lifetime

    %% Progress indicator
    if mod(i, 1000) == 0
        fprintf('%d ', i);
    end
end

elapsed_time = toc;
fprintf('\n\nData generation completed in %.1f seconds (%.2f samples/sec)\n', ...
    elapsed_time, config.N_samples / elapsed_time);

%% Replicate IRF for all samples
fprintf('Replicating IRF for all samples...\n');
irf_all = repmat(IRF, config.N_samples, 1);

%% Compute and display statistics
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Data Statistics:\n');
fprintf('-----------------------------------------------------------------\n');

fprintf('Lifetime Components:\n');
for comp = 1:config.tau_Num
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f] ns\n', ...
        min(tau_components(:, comp)), max(tau_components(:, comp)));
    fprintf('    Mean: %.3f ns, Std: %.3f ns\n', ...
        mean(tau_components(:, comp)), std(tau_components(:, comp)));
end

fprintf('\nFraction Components:\n');
for comp = 1:config.tau_Num
    fprintf('  Component %d:\n', comp);
    fprintf('    Range: [%.3f, %.3f]\n', ...
        min(f_components(:, comp)), max(f_components(:, comp)));
    fprintf('    Mean: %.3f, Std: %.3f\n', ...
        mean(f_components(:, comp)), std(f_components(:, comp)));
end

fprintf('\nAverage Lifetime:\n');
fprintf('  Range: [%.3f, %.3f] ns\n', min(tau_ave), max(tau_ave));
fprintf('  Mean: %.3f ns, Std: %.3f ns\n', mean(tau_ave), std(tau_ave));

fprintf('\nPhoton Counts per Decay:\n');
photon_counts = sum(y, 2);
fprintf('  Range: [%.0f, %.0f]\n', min(photon_counts), max(photon_counts));
fprintf('  Mean: %.1f, Std: %.1f\n', mean(photon_counts), std(photon_counts));

%% Verify fraction sums
fraction_sums = sum(f_components, 2);
fprintf('\nFraction Sum Verification:\n');
fprintf('  Min: %.6f, Max: %.6f\n', min(fraction_sums), max(fraction_sums));
fprintf('  Mean: %.6f (should be 1.000000)\n', mean(fraction_sums));
if max(abs(fraction_sums - 1.0)) > 1e-6
    warning('Fraction sums deviate from 1.0!');
end

%% Save training data
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Saving training data to %s...\n', config.output_file);

save(config.output_file, 'y', 'irf_all', 'tau_components', 'f_components', ...
     'tau_ave', 'config', '-v7.3');

fprintf('Training data saved successfully!\n');

%% Display data structure
fprintf('\n-----------------------------------------------------------------\n');
fprintf('Output Data Structure:\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Variable Name       | Size                        | Description\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('y                   | [%d, %d]    | Fluorescence decay histograms\n', ...
    size(y, 1), size(y, 2));
fprintf('irf_all             | [%d, %d]    | Instrument response functions\n', ...
    size(irf_all, 1), size(irf_all, 2));
fprintf('tau_components      | [%d, %d]        | Individual lifetime values\n', ...
    size(tau_components, 1), size(tau_components, 2));
fprintf('f_components        | [%d, %d]        | Individual fraction values\n', ...
    size(f_components, 1), size(f_components, 2));
fprintf('tau_ave             | [%d, %d]        | Average lifetimes\n', ...
    size(tau_ave, 1), size(tau_ave, 2));
fprintf('config              | struct                      | Configuration settings\n');
fprintf('-----------------------------------------------------------------\n');

%% Create visualization (optional)
create_visualization = true;

if create_visualization
    fprintf('\nCreating visualization...\n');

    figure('Position', [100, 100, 1200, 800]);

    % Plot 1: Sample decays
    subplot(2, 3, 1);
    plot_indices = randsample(config.N_samples, min(10, config.N_samples));
    for i = 1:length(plot_indices)
        idx = plot_indices(i);
        plot(y(idx, :), 'LineWidth', 0.5);
        hold on;
    end
    xlabel('Time Bin');
    ylabel('Photon Counts');
    title('Sample Fluorescence Decays');
    grid on;

    % Plot 2: Lifetime distribution
    subplot(2, 3, 2);
    for comp = 1:config.tau_Num
        histogram(tau_components(:, comp), 50, 'FaceAlpha', 0.5, ...
            'DisplayName', sprintf('Component %d', comp));
        hold on;
    end
    xlabel('Lifetime (ns)');
    ylabel('Count');
    title('Lifetime Component Distributions');
    legend('Location', 'best');
    grid on;

    % Plot 3: Fraction distribution
    subplot(2, 3, 3);
    for comp = 1:config.tau_Num
        histogram(f_components(:, comp), 50, 'FaceAlpha', 0.5, ...
            'DisplayName', sprintf('Component %d', comp));
        hold on;
    end
    xlabel('Fraction');
    ylabel('Count');
    title('Fraction Component Distributions');
    legend('Location', 'best');
    grid on;

    % Plot 4: Average lifetime distribution
    subplot(2, 3, 4);
    histogram(tau_ave, 50, 'FaceColor', [0.3, 0.6, 0.9]);
    xlabel('Average Lifetime (ns)');
    ylabel('Count');
    title('Average Lifetime Distribution');
    grid on;

    % Plot 5: Component correlation
    subplot(2, 3, 5);
    if config.tau_Num == 2
        scatter(tau_components(:, 1), tau_components(:, 2), 10, 'filled', 'MarkerFaceAlpha', 0.3);
        xlabel('Lifetime Component 1 (ns)');
        ylabel('Lifetime Component 2 (ns)');
        title('Lifetime Component Correlation');
    else
        scatter3(tau_components(:, 1), tau_components(:, 2), tau_components(:, 3), ...
            10, 'filled', 'MarkerFaceAlpha', 0.3);
        xlabel('Component 1 (ns)');
        ylabel('Component 2 (ns)');
        zlabel('Component 3 (ns)');
        title('3-Component Lifetime Space');
    end
    grid on;

    % Plot 6: Photon count distribution
    subplot(2, 3, 6);
    histogram(photon_counts, 50, 'FaceColor', [0.9, 0.6, 0.3]);
    xlabel('Total Photon Count');
    ylabel('Count');
    title('Photon Count Distribution');
    grid on;

    sgtitle(sprintf('Multi-Exponential Training Data (%d components, %d samples)', ...
        config.tau_Num, config.N_samples), 'FontSize', 14, 'FontWeight', 'bold');

    % Save figure
    fig_name = strrep(config.output_file, '.mat', '_visualization.png');
    saveas(gcf, fig_name);
    fprintf('Visualization saved to %s\n', fig_name);
end

%% Final summary
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('Next steps:\n');
fprintf('1. Copy %s to the LLE training directory\n', config.output_file);
fprintf('2. Run: python Training_LLE_MultiExp.py\n');
fprintf('3. Update config[''data_name''] = ''%s''\n', config.output_file);
fprintf('=================================================================\n');
