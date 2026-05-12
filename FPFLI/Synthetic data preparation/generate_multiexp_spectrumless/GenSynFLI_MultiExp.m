function GenSynFLI_MultiExp(ind, f_components, RGB_channel, path)
%% GenSynFLI_MultiExp - Generate Multi-Exponential Synthetic FLIM Images (SPECTRUMLESS VERSION)
%
% This function generates synthetic FLIM images with multi-exponential decay
% components using FIXED lifetime values (0.4 ns and 4.5 ns) with NO spectral variation.
% This version is designed for testing training with consistent lifetime components
% but varying mixing fractions.
%
% Inputs:
%   ind           - Sample index for file naming
%   f_components  - Cell array of fraction values for each channel
%                   e.g., {[f1_ch1, f2_ch1], [f1_ch2, f2_ch2]}
%                   Each set of fractions should sum to 1
%   RGB_channel   - RGB channel indices [1,2,3,4]
%   path          - Output path for saved data
%
% Outputs (saved to .mat file):
%   Hist                - Combined fluorescence decay histogram [H, W, 256]
%   tau_gt_components   - Ground truth lifetime components [H, W, 2] (always 0.4 and 4.5)
%   f_gt_components     - Ground truth fraction components [H, W, 2]
%   tau_gt_avg          - Average lifetime map [H, W]
%   Int                 - Total intensity map [H, W]
%   component_masks     - Masks indicating which components are active [H, W, 2]
%
% Example:
%   f = {[0.6, 0.4], [0.7, 0.3]};  % Fractions only (tau fixed at [0.4, 4.5])
%   RGB_channel = [1, 4];
%   GenSynFLI_MultiExp(1, f, RGB_channel, './output/');
%
% Author: Modified for spectrumless multi-exponential prediction
% Date: 2025-12-07

%% FIXED TAU VALUES (NO SPECTRAL VARIATION)
FIXED_TAU_1 = 0.4;   % Short lifetime component (ns)
FIXED_TAU_2 = 4.5;   % Long lifetime component (ns)

%% Validate inputs
N = length(RGB_channel);
assert(length(f_components) == N, ...
    'f_components must match number of channels');

% Validate that fractions sum to 1 for each channel
for i = 1:N
    frac_sum = sum(f_components{i});
    assert(abs(frac_sum - 1.0) < 1e-6, ...
        sprintf('Fractions for channel %d must sum to 1 (got %.6f)', i, frac_sum));
    assert(length(f_components{i}) == 2, ...
        sprintf('Fraction array must have exactly 2 components for channel %d', i));
end

%% Configuration
Size = 256;        % Image size
bin_Num = 256;     % Number of time bins
thres = 20;        % Lower intensity threshold
h = 0.039;         % Bin width (nanoseconds)
IRF = IRF_gaussian(14, h, 0.1673);  % Generate IRF

% Fixed: always 2 components
max_components = 2;
tau_components = cell(N, 1);
for i = 1:N
    tau_components{i} = [FIXED_TAU_1, FIXED_TAU_2];  % Fixed tau values
end

%% Load HPA images
a = dir('*.png');

% Initialize storage
[x, xo, x_1d] = deal(cell(N, 1));
[tau_map_components, f_map_components] = deal(cell(N, 1));
[hist_1d, hist] = deal(cell(N, 1));

%% Process each channel
for i = 1:N
    % Load and resize image
    x{i} = imread(a(ind*4 + RGB_channel(i)).name);
    x{i} = double(imresize(x{i}, [Size Size], 'bicubic'));
    xo{i} = x{i};  % Original intensity

    % Normalize intensity
    M = mean(x{i}(x{i} > thres), 'all');
    Ave = randi(4) + 6;
    x{i} = round(x{i} / (M + 1e-6) * Ave);  % GT intensity
    x_1d{i} = x{i}(:);

    % Initialize component maps (always 2 components with fixed tau)
    tau_map_components{i} = zeros(Size*Size, max_components);
    f_map_components{i} = zeros(Size*Size, max_components);
    hist_1d{i} = zeros(Size*Size, bin_Num);

    % Store FIXED lifetime and user-provided fraction components
    tau_map_components{i}(:, 1) = FIXED_TAU_1;
    tau_map_components{i}(:, 2) = FIXED_TAU_2;
    f_map_components{i}(:, 1) = f_components{i}(1);
    f_map_components{i}(:, 2) = f_components{i}(2);
end

%% Generate temporal decays for each pixel
fprintf('Generating multi-exponential decays (SPECTRUMLESS: tau=[%.1f, %.1f] ns) for sample %d...\n', ...
    FIXED_TAU_1, FIXED_TAU_2, ind);

Hist = zeros([Size, Size, bin_Num]);

for i = 1:N
    fprintf('  Channel %d/%d (RGB %d) with fixed tau=[%.1f, %.1f] ns...\n', ...
        i, N, RGB_channel(i), FIXED_TAU_1, FIXED_TAU_2);

    for j = 1:Size*Size
        if (x_1d{i}(j) > 0)
            % Generate Poisson random number without Statistics Toolbox
            lambda = x_1d{i}(j);
            if exist('poissrnd', 'file')
                N_photons = poissrnd(lambda);
            else
                % Simple Poisson approximation for moderate lambda
                if lambda < 50
                    % Use Knuth's algorithm for small lambda
                    L = exp(-lambda);
                    k = 0;
                    p = 1;
                    while p > L
                        k = k + 1;
                        p = p * rand();
                    end
                    N_photons = k - 1;
                else
                    % For large lambda, use normal approximation
                    N_photons = round(lambda + sqrt(lambda) * randn());
                    N_photons = max(0, N_photons);  % Ensure non-negative
                end
            end

            % Use the multi-exponential decay function with FIXED tau
            hist_1d{i}(j,:) = Fluorescence_multi_decay_nonhomopp(...
                2, ...  % Always 2 components
                [FIXED_TAU_1, FIXED_TAU_2], ...  % Fixed tau values
                f_components{i}, ...  % User-provided fractions
                N_photons, ...
                h, ...
                IRF);
        else
            hist_1d{i}(j,:) = zeros(1, bin_Num);
            % Set components to zero for zero-intensity pixels
            tau_map_components{i}(j, :) = 0;
            f_map_components{i}(j, :) = 0;
        end
    end

    % Reshape to image format
    hist{i} = reshape(hist_1d{i}, [Size, Size, bin_Num]);
    Hist = Hist + hist{i};
end

%% Combine lifetime and fraction components across channels
% Initialize combined component maps
tau_gt_components = zeros([Size, Size, max_components]);
f_gt_components = zeros([Size, Size, max_components]);
component_masks = zeros([Size, Size, max_components]);

% Calculate total intensity for weighted combination
Int = zeros([Size, Size]);
for i = 1:N
    Int = Int + x{i};
end

% Combine components from all channels (intensity-weighted)
fprintf('Combining components across channels...\n');

for comp = 1:max_components
    tau_comp_map = zeros([Size, Size]);
    f_comp_map = zeros([Size, Size]);
    weight_map = zeros([Size, Size]);

    for i = 1:N
        % Reshape component maps
        tau_map_2d = reshape(tau_map_components{i}(:, comp), [Size, Size]);
        f_map_2d = reshape(f_map_components{i}(:, comp), [Size, Size]);

        % Weight by intensity of this channel
        weight = x{i} ./ (Int + 1e-8);

        % Accumulate weighted components
        tau_comp_map = tau_comp_map + weight .* tau_map_2d;
        f_comp_map = f_comp_map + weight .* f_map_2d;
        weight_map = weight_map + weight;

        % Mark component as active
        component_masks(:, :, comp) = component_masks(:, :, comp) + (x{i} > 0);
    end

    % Normalize by total weight
    tau_gt_components(:, :, comp) = tau_comp_map;
    f_gt_components(:, :, comp) = f_comp_map;
end

% Normalize component masks (binary: 0 or 1)
component_masks = component_masks > 0;

% Re-normalize fractions to sum to 1 at each pixel
fprintf('Normalizing fraction components...\n');
fraction_sums = sum(f_gt_components, 3);
for comp = 1:max_components
    f_gt_components(:, :, comp) = f_gt_components(:, :, comp) ./ (fraction_sums + 1e-8);
end

% Set fractions to 0 where there's no intensity
zero_intensity_mask = (Int == 0);
for comp = 1:max_components
    f_gt_components(:, :, comp) = f_gt_components(:, :, comp) .* (~zero_intensity_mask);
    tau_gt_components(:, :, comp) = tau_gt_components(:, :, comp) .* (~zero_intensity_mask);
end

%% Calculate average lifetime (for comparison with standard FPFLI)
tau_gt_avg = zeros([Size, Size]);
for comp = 1:max_components
    tau_gt_avg = tau_gt_avg + tau_gt_components(:, :, comp) .* f_gt_components(:, :, comp);
end

%% Generate output filename
name = ['Sample_', num2str(ind), '_MultiExp_Spectrumless_'];
for i = 1:N
    switch RGB_channel(i)
        case 1
            suffix = 'C1';
        case 2
            suffix = 'C2';
        case 3
            suffix = 'C3';
        case 4
            suffix = 'C4';
    end
    name = append(name, suffix);
end

%% Save data
fprintf('Saving multi-exponential FLIM data (spectrumless)...\n');
save([path, name], 'Hist', 'tau_gt_components', 'f_gt_components', ...
     'tau_gt_avg', 'Int', 'component_masks', 'max_components', '-v7.3');

%% Print statistics
fprintf('Multi-exponential FLIM data generated (SPECTRUMLESS):\n');
fprintf('  Fixed tau values: [%.1f, %.1f] ns\n', FIXED_TAU_1, FIXED_TAU_2);
fprintf('  Total intensity range: [%.1f, %.1f]\n', min(Int(:)), max(Int(:)));
fprintf('  Average lifetime range: [%.3f, %.3f] ns\n', ...
    min(tau_gt_avg(Int>0)), max(tau_gt_avg(Int>0)));

for comp = 1:max_components
    active_pixels = component_masks(:, :, comp);
    if any(active_pixels(:))
        tau_comp = tau_gt_components(:, :, comp);
        f_comp = f_gt_components(:, :, comp);

        fprintf('  Component %d:\n', comp);
        fprintf('    Active pixels: %d (%.1f%%)\n', ...
            sum(active_pixels(:)), 100*sum(active_pixels(:))/(Size*Size));
        fprintf('    Lifetime: %.3f ns (FIXED)\n', ...
            mean(tau_comp(active_pixels)));
        fprintf('    Fraction range: [%.3f, %.3f]\n', ...
            min(f_comp(active_pixels)), max(f_comp(active_pixels)));
        fprintf('    Mean fraction: %.3f\n', mean(f_comp(active_pixels)));
    end
end

fprintf('Saved to: %s\n', [path, name, '.mat']);

end
