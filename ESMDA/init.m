function[P_true,obstrue,TRUE,Perm_obs,IND_2d] = init(casenum,path,num_well)
% clear all; close all;
%%
% diary off;
% matlab.engine.shareEngine
%%
% cd("..")
cd('..\mrst-2018b')
% cd('D:\USC_google_drive\PTE599\HW\PTE573-HW5\mrst-2018b')
startup
% mrstVerbose(0)
cd('..\Pumping')
%% set reservior
nx = 64;                                                                   % number of blocks in each direction
ny = 64;
nz = 1; 
rng(12);

load test_ 
test_cur = test_(casenum,:,:);
test_cur(test_cur<50) = 0;
test_cur(test_cur>=50) = 1;
test_cur = squeeze(test_cur);
TRUE = reshape(test_cur,[64*64,1]);

% plot true test 

%%
% 3 pos
A =[8, 32, 56]; 

% 4 pos

B =[8, 24, 40, 56]; 

switch num_well
    case 9
        [m,n] = ndgrid(A,A);
    case 12
        [m,n] = ndgrid(A,B);
    case 16
        [m,n] = ndgrid(B,B);

end
% [m,n] = ndgrid(A,A);
IND_2d = [m(:),n(:)];
% 

IND = sub2ind([64,64],IND_2d(:,1),IND_2d(:,2));
%% sample observations
OBS_NUM =size(IND_2d,1);                                                       %%%% NUMBER OF OBSERVATIONS
% IND = [20-OBS_NUM:fix(64*64/OBS_NUM):nx*ny];

Perm_obs = TRUE(IND);

%% plot true map
figure('Position',[50 50 400 300])
imagesc(test_cur);
caxis([-0.1, 1.1]);
colormap('jet')
h = colorbar;
h.FontSize = 11;
hold on

% plot obs

save IND1 IND IND_2d
for i =1:OBS_NUM
hold on
I = IND_2d(i,1);
J = IND_2d(i,2);
str = ['M'+string(i)];
h=text(I-3,J+4,str,'FoNTsize',12,'FoNTWeight','bold','Color',"w");
hold on
scatter(I,J,200,'w','x','LineWidth', 3);
title('Log-Permeability','FoNTsize',18)
end

% plot injection well

I = 32;             %% (I,J) index of pumping well
J = 32;
str = ['P','1'];
h=text(I+3.5,J,str,'FoNTsize',12,'FoNTWeight','bold','Color',"red");
axis tight;
hold on
scatter(I,J,200,'red','x','LineWidth', 3);

ax = gca;

% Set the font to Times New Roman
set(ax, 'FontName', 'Times New Roman');
temp_path = append(path,'/True.png');
print(temp_path,'-dpng','-r300')
close;

%% forward
True_ = TRUE;
True_(True_<0.5) = -2.3948;
True_(True_>=0.5) = 0.6009;
[P_true] = pressure_calculation(True_);

%%plot 

OBS = P_true(IND);
obstrue = OBS;
save True_data obstrue P_true True_ test_cur

figure(2)
imagesc(reshape(P_true,[64,64])');
colormap('jet')
colorbar
temp_path = append(path,'/PTrue.png');
title('True Pressure')
print(temp_path,'-dpng','-r300')
close all;

