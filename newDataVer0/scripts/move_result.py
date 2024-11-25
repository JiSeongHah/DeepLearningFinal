import shutil
import os

label_lst = [i for i in range(10)]
# noise_lst = [round(0.1*i,1) for i in range(1,11)]
# label_lst = [9]
# noise_lst = [0.4,0.5,0.6,0.7,0.8,0.9]

# data_lst = ['cifar_1','cifar_2','mnist_1','mnist_2','mnist_3']
data_lst = ['cifar_3']

algorithm_lst = ['smoothed_dsvdd','vanilla_dsvdd','ocsvm']
# algorithm_lst = ['denoising_dsvdd']

for algo in algorithm_lst:
    ori_path  = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts2/history/{algo}_image_noised_add_1112_cifar_3_only'
    
    dest_path = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/history/{algo}_image_noised_add_1112_cifar_3_only'

    error_log = []

    for data_type in data_lst:
        for label in label_lst:
            # for noise in noise_lst:
            
            upper_path= os.path.join(
                    ori_path,
                    data_type,
                    f'normal_label_{label}',
                    # f'noise_{noise}'
                )
            
            under_path_lst = os.listdir(
                upper_path
            )
            
            for under_path in under_path_lst:
                
                final_ori_path = os.path.join(
                    upper_path,
                    under_path
                )
                
                final_dest_path = os.path.join(
                    dest_path,
                    data_type,
                    f'normal_label_{label}',
                    # f'noise_{noise}',
                    under_path,
                )
                
                if os.path.exists(final_dest_path):
                    # print(final_dest_path)
                    error_log.append(final_dest_path)
                    
                else:    
                
                    shutil.copytree(
                        final_ori_path,
                        final_dest_path
                    )
                    print('==========================================================================')
                    print(f'origianl_path : {final_ori_path}')
                    print(f'destination path : {final_dest_path}')
                    print('done')
            
    print()
    print()
    print()
    print()
    # for i in error_log:
        # print(f'copy failed : {i}')
    print(len(error_log))
                    
                
