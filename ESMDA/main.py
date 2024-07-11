import numpy as np 
import matplotlib.pyplot as plt
from esmda____ import * 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
from tqdm import tqdm
def write_param(output_folder,well_number):
## write parameters to run models
## styleGAN
    model_dict = {}
    target ={}
    target['model'] = 'styleGAN'
    target['model_path'] = "./Trained_models/Sgan_z4.pkl"
    target['z_dim'] = 4
    target['select_target'] = 1
    target['well_number'] = well_number

    target['case'] = 1
    target["output_path"] = output_folder+target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    model_dict['stylegan_case1'] = target.copy()

    target['case'] = 2
    target["output_path"] = output_folder+target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    model_dict['stylegan_case2'] = target.copy()


    # ##VAE 16
    # target ={}
    # target['model'] = 'VAE'
    # target['model_path'] = "./Trained_models/vae_bn_16_model.pt"
    # target['z_dim'] = 16
    # target['select_target'] = 1
    # target['well_number'] = well_number

    # target['case'] = 1
    # target["output_path"] = output_folder + target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['VAE16_case1'] = target.copy()
    # target['case'] = 2
    # target["output_path"] = output_folder + target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['VAE16_case2'] = target.copy()

    # ## VAE 32
    # target ={}
    # target['model'] = 'VAE'  
    # target['model_path'] = "./Trained_models/vae_bn_32_model.pt"
    # target['z_dim'] = 32
    # target['select_target'] = 1
    # target['well_number'] = well_number

    # target['case'] = 1
    # target["output_path"] = output_folder +target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['VAE32_case1'] = target.copy()

    # target['case'] = 2
    # target["output_path"] = output_folder +target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['VAE32_case2'] = target.copy()


    # # DCGAN
    # target ={}
    # target['model'] = 'DCGAN'
    # target['model_path'] = "./Trained_models/DCGAN_z4.pt"
    # target['z_dim'] = 4
    # target['select_target'] = 1
    # target['well_number'] = well_number

    # target['case'] = 1
    # target["output_path"] = output_folder +target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['DCGAN_case1'] = target.copy()

    # target['case'] = 2
    # target["output_path"] = output_folder +target['model']+"_"+str(target['z_dim'])+"_output/case"+str(target['case'])
    # model_dict['DCGAN_case2'] = target.copy()


    file_path = "input_param.json"
    with open(file_path, 'w') as json_file:
        json.dump(model_dict, json_file)



write_param('./test_output_9point/',9)
file_path = "input_param.json"
with open(file_path, 'r') as json_file:
    model_dict = json.load(json_file)
for item in tqdm(model_dict):
    print(item)
    print("\n")
    esmda= ESMDA(model_dict[item]["output_path"],model_dict[item]["case"],model_dict[item]['well_number'],100)
    esmda.esmda_forward(model_dict[item])

# write_param('./test_output_12point/',12)
# file_path = "input_param.json"
# with open(file_path, 'r') as json_file:
#     model_dict = json.load(json_file)
# for item in tqdm(model_dict):
#     print(item)
#     print("\n")
#     esmda= ESMDA(model_dict[item]["output_path"],model_dict[item]["case"],model_dict[item]['well_number'],100)
#     esmda.esmda_forward(model_dict[item])    

write_param('./test_output_16point/',16)
file_path = "input_param.json"
with open(file_path, 'r') as json_file:
    model_dict = json.load(json_file)
for item in tqdm(model_dict):
    print(item)
    print("\n")
    esmda= ESMDA(model_dict[item]["output_path"],model_dict[item]["case"],model_dict[item]['well_number'],100)
    esmda.esmda_forward(model_dict[item])