clear
clc
close all
 
% Input distance
x = -3:.1:3;
 
% Reward
y = 1*(1-abs(tanh(x)));

% Plot reward function
plot(x,y)
xlabel('Total Distance from Goal (m)')
ylabel('Reward')
title('Reward Function')
grid on

% Save Image
image_folder = 'Images';
image_file = ('Reward_Function.png');
image_save_path = fullfile(image_folder,image_file);
saveas(gcf, image_save_path)