%============================%
% Naive K-medoids clustering %
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
    min_cost = 999999999;
    min_i = 0;
    for i = 1:cnt
        % Skip the group element that we are interested now
        if group_info(i) ~= j
            continue;
        end
        
        % Cost evaluation
        cost = 0;
        for l = 1:cnt
            % Skip the group element that we are interested now
            if group_info(l) ~= j
                continue;
            end
        
            dist = vecnorm(D(l, :) - D(i, :), NORM);
            cost = cost + dist;
        end
        
        % Obtain the point with the minimal cost
        if cost < min_cost
            min_cost = cost;
            min_i = i;
        end
    end
    
    % Update the new representative point of the group
    z(j, :) = D(min_i, :);
end
end

disp("K-Medoids result:");
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


axis([min(x)-1 max(x)+1 min(y)-1 max(y)+1]);
grid on;
hold off;