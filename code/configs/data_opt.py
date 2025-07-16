class data_augmentation():
        ###Augmentaton parameters
        loadSize = 2000 # The original value is 286,  The Image size is 
        fineSize = 2000
        mean =[0.5,0.5,0.5]
        std= [0.3,0.3,0.3]
        scale=(0.64, 1)
        ratio_crop=(4. / 5., 5. / 4.)
        ratio_expand=(1,1.6)
        ratio_noise= 30
        
        ###--------------