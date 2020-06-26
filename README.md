# VGG_CIFAR100
Using VGG16 Architecture for Recomputation of the dense layers for performance improvement of DCNN in CIFAR100 Dataset

This Project is Based on the Paper by Yimin Yang Professor at Lakehead University 
The Link to the Original IEEE Paper https://ieeexplore.ieee.org/abstract/document/8718406

The Same method can be inplemented on differnet datasets such as Scene-15, CALTECH-101, CALTECH-256, CIFAR-10, CIFAR-100, and SUN-397
This Project was run on a 

The proposed method uses Moore-Penrose Inverse to extract the current residual error to each FC layer, which helps in generating well-generalized features. Further, the weights of each FC layers are recomputed according to the Moore-Penrose Inverse
