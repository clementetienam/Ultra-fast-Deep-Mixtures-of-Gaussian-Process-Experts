Ultra-fast-Deep-Mixture-of-Gaussian-Experts:
---------------------------------------------
This paper considers efficient approaches to the computation of maximum a posteriori (MAP)
estimators of Mixture of Expert models which are computed using a maximization-maximization approach.
 by iteratively maximizing the distribution of experts given allocations
 and allocations given experts. 
 Deep neural networks are used for 
 the gating network and sparse Gaussian processes are used for the 
 experts. This combination forms a robust and efficient model which is able
 to significantly outperform competing methods. 
 We also show that a recently introduced method called CCR 
 is capable of providing a good approximation of the optimizer 
 extremely quickly. This approximation can then be further refined 
 with the iterative algorithm.

Getting Started:
---------------------------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.


Prerequisites:
-------------------------------
MATLAB 2015 upwards

Methods:
-------------------------------
Three methods are available for the supervised learning problem;
- CCR: 1 pass Sparse Gaussian process experts with a Deep Neural network gating network
- CCR-MM: Iterative Sparse Gaussian process experts with a Deep Neural network gating network with K-means for initialisation of latent variables 
and inducing points
- MM-MM: Iterative Sparse Gaussian process experts with a Deep Neural network gating network with random initialisation of latent variables 
and K-means for initialising the inducing points


1. Datasets
-----------------------------
The 10 datasets used in the paper are provided in the script, together with an extra 4 datasets.

Running the Numerical Experiment:
Run the script Main.m following the prompts on the screen

2. Dependencies
----------------------------
- CCR,CCR-MM and MM-MM requires the : GPML [12] in addition to my own library of utility functions (CKS/CKS_DNN/CKS_MLP)
- The CKS's ;library contains necessary scripts for visualisation, Running the Neural Network and computing some hard and soft predictions

All libraries are included for your convenience.

3. Manuscript
-----------------------------

4. Extras
--------------------------------------
Extra methods are included also;
- Running supervised learning models with DNN and MLP alone (Requires the netlab and MATLAB DNN tool box)
- Running CCR/CCR-MM and MM-MM with DNN/DNN for the experts ad gates respectively and MLP/DNN for the experts and gates respectively (Requires netlab and MATLAB DNN toolbox)

Author:
--------------------------------
Dr Clement Etienam- Postdoctoral Research Associate, Oak Ridge National Laboratory(ORNL)/School of Mathematics, University of Manchester 

In Collaboration with:
---------------------------------
Professor Kody Law- Postdoctoral Supervisor and Chair of Applied Mathematics at the School of Mathematics, University of Manchester 

Professor Sara Wade- Senior Lecturer in Applied Mathematics & Statistics University of Edinburgh


Acknowledgments:
------------------------------
This work is supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research (ASCR) 
Scientific Discovery through Advanced Computing (SciDAC) project on Advanced Tokamak Modelling (AToM), under field work proposal number ERKJ123.

References:
----------------------------

[1] Luca Ambrogioni, Umut Güçlü, Marcel AJ van Gerven, and Eric Maris. The kernel mixture network: A non-parametric method for conditional density estimation 
of continuous random variables. arXiv preprint arXiv:1705.07111, 2017.

[2] Christopher M Bishop. Mixture density networks. 1994.

[3] Isobel C. Gormley and Sylvia Frühwirth-Schnatter. Mixtures of Experts Models. Chapman and Hall/CRC, 2019.

[4] R.B. Gramacy and H.K. Lee. Bayesian treed Gaussian process models with an application to computer modeling. Journal of the American Statistical Association, 103(483):1119–1130,
2008.

[5] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, Geoffrey E Hinton, et al. Adaptive
mixtures of local experts. Neural computation, 3(1):79–87, 1991.
2

[6] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm.
Neural computation, 6(2):181–214, 1994.

[7] Trung Nguyen and Edwin Bonilla. Fast allocation of gaussian process experts. In International
Conference on Machine Learning, pages 145–153, 2014.

[8] Carl E Rasmussen and Zoubin Ghahramani. Infinite mixtures of gaussian process experts. In
Advances in neural information processing systems, pages 881–888, 2002.

[9] Tommaso Rigon and Daniele Durante. Tractable bayesian density regression via logit stickbreaking
priors. arXiv preprint arXiv:1701.02969, 2017.

[10] Volker Tresp. Mixtures of gaussian processes. In Advances in neural information processing
systems, pages 654–660, 2001.

[11] Lei Xu, Michael I Jordan, and Geoffrey E Hinton. An alternative model for mixtures of experts.

[12] Rasmussen, Carl Edward and Nickisch, Hannes. Gaussian processes for machine learning (gpml) toolbox. The
Journal of Machine Learning Research, 11:3011–3015, 2010

[13] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and
Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 
1(2639-8001-2019-4-491):491, 2019.
