%============================%
% Naive K-medians clustering %
%============================%

NORM = 1; % L1 or L2 norm
MAX_ITERATION = 500;

D = [0, -6;
     4, 4;
     0, 0;
     -5, 2];
k = 2;
cnt = size(D, 1);
cost = 999999999;
group_info = zeros(cnt, 1);
iterations = 0;

% Step1: Random select the representative points
%z = zeros(k, size(D, 2));
%D_rand = D;
%left_cnt = cnt;
%for i = 1:k
%    samp = randi(left_cnt);
%    z(i, :) = D_rand(samp, :);
%    D_rand = [D_rand(1:samp-1, :); D_rand(samp+1:end, :)];
%    left_cnt = size(D_rand, 1);
%end

% Step1: Random select the representative points
z(1, :) = [-5, 2];
z(2, :) = [0, -6];

while iterations < MAX_ITERATION
iterations = iterations + 1;
%disp("Iteration:");
%disp(iterations);

% Step2-1: Select group for all points that has minimum distance
for i = 1:cnt
    min_dist = 999999999;
    min_k = 0;
    for j = 1:k
        dist = vecnorm(D(i, :) - z(j, :), NORM);
        if dist < min_dist
            min_dist = dist;
            min_k = j;
        end
    end
    cost = cost + min_dist;
    group_info(i) = min_k;
end

% Step2-2: Find the best representatives of all groups
for j = 1:k    
    % Obtain group members
    M = [];
    for i = 1:cnt
        % Skip the group element that we are interested now
        if group_info(i) ~= j
            continue;
        end
        
        M = [M; D(i, :)];
    end
    
    % Find the centroid with median
    med_x = median(M(:, 1));
    med_y = median(M(:, 2));
    z(j, :) = [med_x, med_y];
end
end

disp("K-Medians result:");
disp(group_info);
disp("z (representatives)");
disp(z)

%=====================%
% Plotting the result %
%=====================%

x = D(:, 1);
y = D(:, 2);

figure;

% Loop through each row and plot the points with different colors based on group_info
hold on;
for i = 1:length(group_info)
    if group_info(i) == 1
        plot(x(i), y(i), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Red for group 1
    elseif group_info(i) == 2
        plot(x(i), y(i), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % Blue for group 2
    end
end

cent_x = z(:, 1);
cent_y = z(:, 2);

for i = 1:k
    plot(cent_x(i), cent_y(i), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Green for centroid
end

axis([min(x)-1 max(x)+1 min(y)-1 max(y)+1]);
grid on;
hold off;
