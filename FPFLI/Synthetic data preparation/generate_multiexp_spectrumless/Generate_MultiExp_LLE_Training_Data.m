%% Generate Multi-Exponential LLE Training Data (SPECTRUMLESS VERSION)
% Creates training data for multi-exponential Local Lifetime Estimation
% with FIXED lifetime values (0.4 ns and 4.5 ns) and NO spectral variation.
%
% Output: Training_data_multiexp_spectrumless.mat containing:
%   - y: Fluorescence decay histograms [N_samples, 256]
%   - irf: Instrument response functions [N_samples, 256]
%   - tau_components: Individual lifetime values [N_samples, 2] (always [0.4, 4.5])
%   - f_components: Individual fraction values [N_samples, 2]
%   - tau_ave: Average lifetimes [N_samples, 1]
%
% Author: MG
% Date: 2025-12-07

clear; clc;

%% FIXED TAU VALUES (NO SPECTRAL VARIATION)
FIXED_TAU_1 = 0.4;   % Short lifetime component (ns)
FIXED_TAU_2 = 4.5;   % Long lifetime component (ns)

%% Configuration
config.tau_Num = 2;              % Always 2 components (fixed)
config.N_samples = 10000;        % Number of training samples
config.bin_width = 0.039;        % Time bin width in nanoseconds
config.bin_Num = 256;            % Number of time bins
config.output_file = 'Training_data_multiexp_spectrumless.mat';

% Fixed lifetime values (NO VARIATION)
config.tau_fixed = [FIXED_TAU_1, FIXED_TAU_2];

% Photon count range (varying to simulate different intensities)
config.N_photon_min = 100;
config.N_photon_max = 500;

fprintf('=================================================================\n');
fprintf('Multi-Exponential LLE Training Data Generation (SPECTRUMLESS)\n');
fprintf('=================================================================\n');
fprintf('Configuration:\n');
fprintf('  Number of components: %d (FIXED)\n', config.tau_Num);
fprintf('  Fixed tau values: [%.1f, %.1f] ns (NO VARIATION)\n', FIXED_TAU_1, FIXED_TAU_2);
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
fprintf('\nGenerating %d multi-exponential decay samples with FIXED tau=[%.1f, %.1f] ns...\n', ...
    config.N_samples, FIXED_TAU_1, FIXED_TAU_2);
fprintf('Progress: ');

tic;
for i = 1:config.N_samples
    %% Use FIXED lifetime components (NO VARIATION)
    tau = config.tau_fixed;  % Always [0.4, 4.5]

    %% Generate random fraction components
    % Method: Dirichlet distribution (uniform over simplex)
    % This ensures fractions are positive and sum to 1
    % Using exponential random variables (no Statistics Toolbox needed)
    % Dirichlet(alpha) where alpha = [1,1,...] is equivalent to
    % normalizing independent Exp(1) random variables
    exp_rvs = -log(rand(1, config.tau_Num));  % Exponential(1) samples
    f = exp_rvs / sum(exp_rvs);  % Normalize to sum to 1

    %% Generate photon count (varying intensity)
    N_photons = config.N_photon_min + randi(config.N_photon_max - config.N_photon_min);

    %% Generate fluorescence decay histogram
    y(i,:) = Fluorescence_multi_decay_nonhomopp(...
        config.tau_Num, tau, f, N_photons, config.bin_width, IRF);

    %% Store ground truth components
    tau_components(i,:) = tau;  % All samples have same tau
    f_components(i,:) = f;      % Fractions vary
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

fprintf('Lifetime Components (FIXED - NO VARIATION):\n');
for comp = 1:config.tau_Num
    fprintf('  Component %d:\n', comp);
    fprintf('    Value: %.3f ns (FIXED)\n', config.tau_fixed(comp));
    fprintf('    Range: [%.3f, %.3f] ns (should be constant)\n', ...
        min(tau_components(:, comp)), max(tau_components(:, comp)));
    fprintf('    Mean: %.3f ns, Std: %.6f ns (std should be ~0)\n', ...
        mean(tau_components(:, comp)), std(tau_components(:, comp)));
end

fprintf('\nFraction Components (VARYING):\n');
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
fprintf('  Note: Average lifetime varies only due to fraction variation\n');

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
fprintf('tau_components      | [%d, %d]        | Fixed lifetime values (all [%.1f, %.1f])\n', ...
    size(tau_components, 1), size(tau_components, 2), FIXED_TAU_1, FIXED_TAU_2);
fprintf('f_components        | [%d, %d]        | Individual fraction values (varying)\n', ...
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
    % Random sample without Statistics Toolbox
    n_plot = min(10, config.N_samples);
    plot_indices = randperm(config.N_samples, n_plot);
    for i = 1:length(plot_indices)
        idx = plot_indices(i);
        plot(y(idx, :), 'LineWidth', 0.5);
        hold on;
    end
    xlabel('Time Bin');
    ylabel('Photon Counts');
    title('Sample Fluorescence Decays');
    grid on;

    % Plot 2: Lifetime distribution (should be constant)
    subplot(2, 3, 2);
    for comp = 1:config.tau_Num
        histogram(tau_components(:, comp), 50, 'FaceAlpha', 0.5, ...
            'DisplayName', sprintf('Component %d (%.1f ns)', comp, config.tau_fixed(comp)));
        hold on;
    end
    xlabel('Lifetime (ns)');
    ylabel('Count');
    title('Lifetime Component Distributions (FIXED)');
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
    title('Fraction Component Distributions (VARYING)');
    legend('Location', 'best');
    grid on;

    % Plot 4: Average lifetime distribution
    subplot(2, 3, 4);
    histogram(tau_ave, 50, 'FaceColor', [0.3, 0.6, 0.9]);
    xlabel('Average Lifetime (ns)');
    ylabel('Count');
    title('Average Lifetime Distribution');
    grid on;

    % Plot 5: Fraction space (2D simplex)
    subplot(2, 3, 5);
    scatter(f_components(:, 1), f_components(:, 2), 10, 'filled', 'MarkerFaceAlpha', 0.3);
    xlabel(sprintf('Fraction Component 1 (tau=%.1f ns)', FIXED_TAU_1));
    ylabel(sprintf('Fraction Component 2 (tau=%.1f ns)', FIXED_TAU_2));
    title('Fraction Space (2D Simplex)');
    xlim([0, 1]);
    ylim([0, 1]);
    % Add constraint line f1 + f2 = 1
    hold on;
    plot([0, 1], [1, 0], 'r--', 'LineWidth', 2, 'DisplayName', 'f1 + f2 = 1');
    legend('Location', 'best');
    grid on;

    % Plot 6: Photon count distribution
    subplot(2, 3, 6);
    histogram(photon_counts, 50, 'FaceColor', [0.9, 0.6, 0.3]);
    xlabel('Total Photon Count');
    ylabel('Count');
    title('Photon Count Distribution');
    grid on;

    sgtitle(sprintf('Multi-Exponential Training Data (SPECTRUMLESS: tau=[%.1f, %.1f] ns, %d samples)', ...
        FIXED_TAU_1, FIXED_TAU_2, config.N_samples), 'FontSize', 14, 'FontWeight', 'bold');

    % Save figure
    fig_name = strrep(config.output_file, '.mat', '_visualization.png');
    saveas(gcf, fig_name);
    fprintf('Visualization saved to %s\n', fig_name);
end

%% Final summary
fprintf('\n=================================================================\n');
fprintf('DATA GENERATION COMPLETE (SPECTRUMLESS)!\n');
fprintf('=================================================================\n');
fprintf('Key features:\n');
fprintf('  - Fixed lifetime values: [%.1f, %.1f] ns (NO spectral variation)\n', ...
    FIXED_TAU_1, FIXED_TAU_2);
fprintf('  - Varying fractions: Models different mixing ratios\n');
fprintf('  - %d training samples generated\n', config.N_samples);
fprintf('\nNext steps:\n');
fprintf('1. Copy %s to the LLE training directory\n', config.output_file);
fprintf('2. Run: python Training_LLE_MultiExp.py\n');
fprintf('3. Update config[''data_name''] = ''%s''\n', config.output_file);
fprintf('=================================================================\n');
