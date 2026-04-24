% 复现实验结果图。

% 模型标签与 README 表格保持一致；长标签使用换行来逼近论文/汇报图中的排版效果。
model_labels = {
    'GPT-4o-mini'
    'Claude 3.5 Sonnet'
    'Claude 3.7 Sonnet\newline(8K budget)'
    sprintf('Instruct')
    sprintf('Instruct+GRPO')
    sprintf('Instruct+SFT+GRPO')
};

% 两组柱子的数值分别对应：成功通关数、以及“只统计获胜局”的平均猜词次数。
% 附件图使用单 Y 轴即可容纳这两组值，因此这里保持同样的可视化方式而不改成双轴图。
solved_games = [1, 8, 10, 0, 3, 7];
avg_guesses = [4, 4, 3.9, 6, 4, 4];
plot_data = [solved_games; avg_guesses]';

num_groups = size(plot_data, 1);
num_series = size(plot_data, 2);

fig = figure('Color', 'w', 'Position', [100, 100, 1200, 560]);
ax = axes('Parent', fig, 'Position', [0.08, 0.22, 0.80, 0.66]);

bars = bar(ax, plot_data, 'grouped', 'BarWidth', 0.82);
bars(1).FaceColor = [72, 126, 160] / 255;
bars(2).FaceColor = [7, 53, 79] / 255;

ax.FontName = 'Arial';
ax.FontSize = 12;
ax.LineWidth = 1;
ax.Box = 'off';
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridColor = [0.75, 0.78, 0.82];
ax.GridAlpha = 1;
ax.Layer = 'bottom';
ax.XTick = 1:num_groups;
ax.XTickLabel = model_labels;
ax.TickLabelInterpreter = 'tex';
ax.TickLength = [0, 0];
ax.YLim = [0, 11.5];
ax.XLim = [0.4, num_groups + 0.6];

lg = legend( ...
    ax, ...
    {'Solved Games', 'Avg # Guesses (In solved games)'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'Box', 'off' ...
);
lg.FontSize = 13;
if isprop(lg, 'ItemTokenSize')
    lg.ItemTokenSize = [24, 12];
end

% 隐藏左侧 y 轴线，但保留横向网格线；y 轴数字改为手工绘制，避免随轴线一起消失。
y_ticks = ax.YTick;
ax.YColor = fig.Color;
ax.YTickLabel = [];

% 适度缩小左移偏移量，让 y 轴数字更贴近横向网格线和绘图区。
label_x = ax.XLim(1) - 0.005 * diff(ax.XLim);
for tick_idx = 1:numel(y_ticks)
    text( ...
        ax, ...
        label_x, ...
        y_ticks(tick_idx), ...
        num2str(y_ticks(tick_idx), '%g'), ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', ax.FontSize, ...
        'Color', [0.2, 0.2, 0.2], ...
        'Clipping', 'off' ...
    );
end

% 柱顶数值标注与附件图保持一致，便于直接阅读，不需要再回表格查值。
for series_idx = 1:num_series
    x_positions = local_bar_centers(bars(series_idx), num_groups, series_idx, num_series);
    y_values = bars(series_idx).YData;

    for group_idx = 1:num_groups
        label_y = y_values(group_idx) + 0.35;
        text( ...
            ax, ...
            x_positions(group_idx), ...
            label_y, ...
            num2str(y_values(group_idx), '%g'), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', bars(series_idx).FaceColor ...
        );
    end
end

% 如需导出为图片，可取消下面这行的注释。
% exportgraphics(fig, 'benchmark_result_reproduction.png', 'Resolution', 300);


function x_positions = local_bar_centers(bar_handle, num_groups, series_idx, num_series)
% 返回每根柱子的 x 中心点。
% 新版 MATLAB 可直接读取 XEndPoints；若运行环境较老，则回退到官方 grouped bar 近似公式。
    if isprop(bar_handle, 'XEndPoints') && ~isempty(bar_handle.XEndPoints)
        x_positions = bar_handle.XEndPoints;
        return;
    end

    group_width = min(0.8, num_series / (num_series + 1.5));
    group_centers = 1:num_groups;
    x_offset = (series_idx - (num_series + 1) / 2) * group_width / num_series;
    x_positions = group_centers + x_offset;
end
