# Image-Translation-Problem-with-ModGAN
## <a href = "">Link</a> for the demo.
## Motivation & Problem Statement
* Face generation using GANs or Generative Adversarial Networks is highly random,
  as every forward pass generates an image from a randomly sampled noise vector.

* Most GANs focus only on the resolution aspect, without paying much attention 
  to facial attributes like skin color, hair color, etc.

* Thus we aim to control the generated output by providing auxiliary information 
  in the form of an attribute vector or a rough sketch.
  
* The assumption is that attributes will serve as a guiding signal to the generative 
  model and help it determine the corresponding output representation.

* The image translation problem, i.e. generation of realistic output images from 
  rough images or outlines has been explored for paired and unpaired image data.

## Pipeline
![System Pipeline](pipeline.png)

## How to run?
```
  1. Create the conda environment on your system using the 'environment.yml' file using the following command:
     'conda env create -f environment.yml'.
  2. Make a folder with the name of structure to be reconstructed(eg: tower) in the 'Datasets' directory.
  3. In this folder, add the images of the 3D stucture such that all the images are in an iterative manner 
     as if the camera is moving around the structure in 360 degree with each pair of consecutive images sharing 
     some key points. All images should be captured by the same camera.
  3. Now, add the intrinsic camera matrix of your calibrated camera in a txt file named 'K.txt' and place it in 
     the same stucture named folder(eg: tower) along with the images.
  4. Finally, activate the conda environment and run the command 'python main.py folder_name'. Here, we are 
     passing the name of our structure folder(eg: tower) as the command line input. So, in our example case 
     the command would look like 'python main.py tower'.
  5. The reconstructed 3D structure would be saved in a .ply file named 'reconstruction.py'. This file would be
     saved in a folder in the 'res' directory named same as the input structure(eg: tower). This folder will 
     also contain the reconstruction error plot(plot.png) and the pose array in the file 'pose_array.csv'.
```
