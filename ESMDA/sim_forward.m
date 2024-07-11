function [P_obs,P_maps] = sim_forward(perm_all)
    diary off;
    load IND1;
    P_maps=[];
    P_obs = [];
    [n,~,~] = size(perm_all);
    perm_all = double(perm_all);
    for i = 1:n
        perm_map = reshape(perm_all(i,:,:),[64*64,1]);
%         perm_map = perm_map*(0.6009+2.3948)-2.3948;
        perm_map(perm_map<0.5) = -2.3948;
        perm_map(perm_map>=0.5) = 0.6009;
        P_map = pressure_calculation(perm_map);
        P_cur = P_map(IND);
        %
        P_maps = [P_maps,P_map];
        P_obs = [P_obs, P_cur];
    
    end
    P_obs = P_obs';
    P_maps = P_maps';
end