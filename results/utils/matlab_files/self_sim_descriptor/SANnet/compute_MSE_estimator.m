clear all
close all
clc

load OUTPUTS %resp_out
load CONTENTS %resp_cont

%20 styles, 44 images (remove style 4 and content 11)

MSE = zeros(19, 43);

for l = 0:19 %iterate on STYLES
    if l == 4
        disp('STYLE 4 SKIPPED')
    else
        for k = 0:43 %iterate on CONTENTS
            disp(['Processing content ', num2str(k), ' in style ', num2str(l)])
            if k == 11
                disp('IMAGE 11 SKIPPED')
            else
                %WE DO OUR THING HERE
                MSE(l+1,k+1) = immse(reshape(resp_cont(l+1,k+1, :,:), [8, 9604]), reshape(resp_out(l+1,k+1, :,:), [8, 9604]));
%                 MSE(l+1,k+1) = mae(reshape(resp_cont(l+1,k+1, :,:), [8, 9604]), reshape(resp_out(l+1,5, :,:), [8, 9604]));
            end
        end
    end
end

MSE(:, 12) = [];
MSE(5, :) = [];

MSE(MSE>0.2) = 0.2;

figure(1)
heatmap(MSE,'Colormap', jet,'GridVisible','off','ColorMethod','mean', 'FontName', 'Times New Roman', 'FontSize', 15)
% heatmap(MSE)
set(gcf,'color','w');
xlabel('Contents')
ylabel('Styles')
title('Self-similarity indicator MSE')
Ax = gca;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));