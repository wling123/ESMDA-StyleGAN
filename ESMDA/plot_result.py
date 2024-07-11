import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from mpl_toolkits.mplot3d import Axes3D

def plot_perm_mean(image_list, loc,path):
    fig, axs = plt.subplots(1,5, figsize=(18, 4))
    axs = axs.flatten()
    for i in range(5):
        im=axs[i].imshow(np.mean(image_list[i],axis =0),cmap='jet',vmin = -0.1, vmax = 1.1,  interpolation='none')
        axs[i].scatter(loc[0,:],loc[1,:],s =100,c='w')
        axs[i].set_title("Iter "+str(i),fontsize=24)
        axs[i].axis('off')

    fig.suptitle("Mean of Log-permeability",fontsize=26)
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(), location = 'right',shrink =1)
    plt.savefig(path+'/perm_mean.png',bbox_inches='tight',dpi=300, transparent=True)
    plt.close()

def plot_perm_std(image_list, loc,path):
    
    fig, axs = plt.subplots(1,5, figsize=(18, 4))
    axs = axs.flatten()
    # axs[0].imshow(Perm_true,cmap='jet',  interpolation='none')
    for i in range(5):
        im = axs[i].imshow(np.std(image_list[i],axis =0),cmap='jet',vmin = -0.1, vmax = 0.5,  interpolation='none')
        axs[i].scatter(loc[0,:],loc[1,:],s =100,c='w')
        axs[i].set_title("Iter "+str(i),fontsize=24)
        axs[i].axis('off')

    fig.suptitle(r"$\mathrm{\sigma}$ of Log-permeability",fontsize=26)
    # plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(), location = 'right',shrink =0.7)
    plt.savefig(path+'/perm_std.png',bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
def rank_performance_mean(data,keep_num = 80):
    
    # data = p_list[-1]  # Shape: (100, 10)

    # Calculate z-scores for each realization
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))  # Shape: (100, 10)

    # Calculate outlier scores for each realization
    outlier_scores = np.max(z_scores, axis=1)  # Shape: (100,)

    # Sort the outlier scores and get the indices of the 80 realizations with the lowest scores
    sorted_indices = np.argsort(outlier_scores)
    selected_indices = sorted_indices[:keep_num]

    # Keep the selected realizations
    selected_data = data[selected_indices]
    return selected_data, selected_indices

def plot_p_well(p_list,p_true, path,p_ind):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax[0].boxplot(p_list[0]/1e3, vert=True)
    ax[0].set_ylabel('Value')
    ax[0].set_title('Box Plot with Reference Points',fontsize=25)
    ax[0].scatter(range(1, len(p_true)+1), p_true/1e3, color='r',label = 'Reference')
    # ax.scatter(range(1, len(p_true)+1), np.mean(p_list[0]/1e3,axis=0), color='b')
    # Show the plot
    ax[0].set_xlabel('Mointering Wells',fontsize=18)
    ax[0].set_ylabel('Pressure (kPa)',fontsize=18)
    ax[0].set_title('Initial',fontsize=20)
    ax[0].legend(fontsize=18)
    #
    ax[1].boxplot(p_list[-1]/1e3, vert=True)
    ax[1].set_ylabel('Value',fontsize=18)
    ax[1].set_title('Box Plot with Reference Points',fontsize=25)
    ax[1].scatter(range(1, len(p_true)+1), p_true/1e3, color='r',label = 'Reference')
    # ax.scatter(range(1, len(p_true)+1), np.mean(p_list[0]/1e3,axis=0), color='b')
    # Show the plot
    ax[1].set_ylabel('Pressure (kPa)',fontsize=18)
    ax[1].set_xlabel('Mointering Wells',fontsize=18)
    ax[1].set_title('Updated',fontsize=20)
    ax[1].legend(fontsize=18)

    plt.suptitle("Pressure at Observation Wells",fontsize=22)
    plt.tight_layout()
    plt.savefig(path+'/well_pressure.png',bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def plot_samples(recon_img,path,name,img_name):
    font = {'family': 'serif', 'size': 12}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.title(name, fontsize=19)
    plt.imshow(torch.mean(recon_img,2),cmap='jet',vmax=1.1,vmin=-.1)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(path+"/"+name+img_name+'.png',bbox_inches='tight')
    plt.close()
def plot_samples_paper(path,image_list,p_ind):
    recon_image = torch.from_numpy(image_list[0][p_ind[:9]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Prior","_heads")

    recon_image = torch.from_numpy(image_list[-1][p_ind[:9]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Posterior","_heads")
##

    recon_image = torch.from_numpy(image_list[0][p_ind[40:40+9]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Prior","_mids")

    recon_image = torch.from_numpy(image_list[-1][p_ind[40:40+9]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Posterior","_mids")

    ## 
    recon_image = torch.from_numpy(image_list[0][p_ind[-9:]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Prior","_tails")

    recon_image = torch.from_numpy(image_list[-1][p_ind[-9:]]).view(-1,1,64,64)
    recon_image[recon_image == 0] = -1
    recon_img = np.transpose(vutils.make_grid(recon_image, nrow=3,padding=5,normalize=False),(1,2,0))
    recon_img[recon_img == 0] = np.nan
    recon_img[recon_img == -1] = 0
    plot_samples(recon_img,path,"Posterior","_tails")

def plot_obj(input,path):

    fig = plt.figure(figsize=(10, 5))
    plt.plot(input,'.-')
    plt.title('Objective function',fontsize=22)
    plt.margins(0) 
    plt.savefig(path+'/obj.png',bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def plot_true_perm_map(input,path):
    permeability_map= input.perm_map_true.reshape([64,64]).T
    permeability_map= np.tile(permeability_map,(1,1,1)).transpose([1,2,0])

    injection = [(32, 32, 2)]
    injection = np.array(injection)
    monitor = np.array([(input.loc[1][i],input.loc[0][i],1) for i in range(input.loc[0].shape[0])])


    data = permeability_map
    data[injection[:, 0],injection[:, 1],:] = 1
    data[monitor[:, 0],monitor[:, 1],:] = 0.5
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a boolean array of the same shape as data to represent filled voxels
    filled = np.ones_like(data, dtype=bool)

    # Plot the 3D cube using voxels
    ax.voxels(filled, edgecolor='none', facecolors=plt.cm.jet(data), linewidth=0)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    # Compress the Z-axis
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.4, 1]))
    plt.axis('off')

    # Show the plot
    # Add points on the top surface

    # Plot the points
    ax.scatter(injection[:, 0], injection[:, 1], injection[:, 2]-0.9,marker='x', color='red', s=120, label='injection well')
    ax.scatter(monitor[:, 0], monitor[:, 1], monitor[:, 2],marker='.', color='green', s=120, label='monitoring well')
    plt.legend(loc='lower left')
    # Add text annotations
    # for point in injection:
    #     ax.text(point[0], point[1], point[2], f'Int', fontsize=10)

    # Show the colorbar to represent the data values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
    sm._A = []  # Hack to make colorbar work with scatter plot
    cax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
    cbar = fig.colorbar(sm, cax=cax, shrink=0.5,pad=0.05)
    # Show the plot

    # plt.show()
    ax.view_init(elev=30, azim=30)
    ##
    plt.savefig(path+'/perm3d.png',bbox_inches='tight', dpi=300, transparent=True)
    plt.close()