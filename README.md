# WoWIconGAN

Our group consisted of Bjarke Larsen, Ethan Osborne, and Marj Cuerdo. To run this code, you need to have tensorflow, keras, cudatoolkit, cudnn, and numpy installed in an anaconda environment. To run this project, you need to run the command python wowgan.py in the command prompt/terminal in the anaconda environment. Also, ask me for the full project in a zipfile because there were too many files to upload to Github.


We created a batch of new icons based on a directory of 4,000+ World of Warcraft images. After not finding luck with the original model, we tweaked it to first train low resolution versions of the images, followed by medium resolution versions, and then finally high resolution images. Below are some results.

Original Icons:
![first image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/OriginalWoWIcons/01.png)![second image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/OriginalWoWIcons/01_1.png)![third image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/OriginalWoWIcons/01_2.png)![fourth image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/OriginalWoWIcons/01_3.png)![fifth image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/OriginalWoWIcons/01_4.png)

First Test Icons:
![firstT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/FirstTryWoWIcons/generated_img_065_0.png)![secondT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/FirstTryWoWIcons/generated_img_065_1.png)![thirdT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/FirstTryWoWIcons/generated_img_065_3.png)![fourthT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/FirstTryWoWIcons/generated_img_065_4.png)![fifthT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/FirstTryWoWIcons/generated_img_066_2.png)

Low Res Test Icons:
![firstLrR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowerResTestIcons/l%200.png)![secondLrR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowerResTestIcons/l%201.png)![thirdLrR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowerResTestIcons/l%202.png)![fourthLrR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowerResTestIcons/l%203.png)![fifthLrR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowerResTestIcons/l%204.png)

Medium Res Test Icons:
![firstLR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResTestIcons/l%20556.png)![secondLR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResTestIcons/l%20557.png)![thirdLR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResTestIcons/l%20558.png)![fourthLR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResTestIcons/l%20559.png)![fifthLR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResTestIcons/l%20560.png)

Low Res Trained Icons:
![firstLT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResWoWIcons/low_generated_img_078_0.png)![secondLT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResWoWIcons/low_generated_img_098_1.png)![thirdLT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResWoWIcons/low_generated_img_066_2.png)![fourthLT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResWoWIcons/low_generated_img_075_2.png)![fifthLT image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/LowResWoWIcons/low_generated_img_099_2.png)

Medium Res Trained Icons:
![firstMR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/MedResWoWIcons/med_generated_img_019_4.png)![secondMR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/MedResWoWIcons/med_generated_img_003_0.png)![thirdMR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/MedResWoWIcons/med_generated_img_008_4.png)![fourthMR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/MedResWoWIcons/med_generated_img_013_0.png)![fifthMR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/MedResWoWIcons/med_generated_img_000_0.png)

High Res Trained Icons:
![firstHR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/HighResWoWIcons/high_generated_img_000_2.png)![secondHR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/HighResWoWIcons/high_generated_img_005_2.png)![thirdHR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/HighResWoWIcons/high_generated_img_010_0.png)![fourthHR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/HighResWoWIcons/high_generated_img_015_4.png)![fifthHR image](https://github.com/ethanlosborne/WoWIconGAN/blob/main/HighResWoWIcons/high_generated_img_019_3.png)

The constraints that we had to define were the limitation of just ~4,000 images that were of various types so we ended with the more amorphous images. 
