# Improving Sparse-View Cone-Beam Tomography with 3D Gaussian Mixture Models
(c) Chang-Chieh Cheng\
Information Technology Service Center\
National Yang Ming Chiao Tung University\
Email: [jameschengcs@nycu.edu.tw](mailto:jameschengcs@nycu.edu.tw)

## Abstract
Sparse-view cone-beam computed tomography (CBCT) image reconstruction remains challenging due to severe streak artifacts and loss of structural fidelity. 
This paper introduces GMMCBCT, a refinement framework that represents volumetric data using three-dimensional Gaussian mixtures and iteratively improves reconstruction quality in the projection domain. 
Starting from an initial volume reconstructed by an analytic or learning-based method, GMMCBCT initializes Gaussian parameters from reprojection errors and refines them through coarse and fine projection refinements. 
The optimized Gaussian mixture is then used to generate the final reconstruction. 
Experiments on a medical dataset demonstrate that GMMCBCT consistently enhances reconstruction quality across different baselines. 
Compared with the baselines, GMMCBCT achieves lower MAE and MSE, and higher SSIM and PSNR, with particularly notable gains for the conventional FDK method. 
Ablation studies further confirm the effectiveness of the proposed initialization strategy and refinement steps. 
Although GMMCBCT entails higher computational cost, it provides a general and flexible framework that improves both analytic and deep-learning-based approaches, offering a promising direction for accurate sparse-view CBCT reconstruction.
