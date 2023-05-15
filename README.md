# EM-Guided Policy Search
## Sample efficient model based reinforcement learning with **Maximum-likelihood**

Guided policy search algorithms have proved to work with incredible accuracy for not only controlling a complicated dynamical system, but also learning optimal policies for different unseen instances. In most of algorithms which have been proposed for policy search and learning, one assumes the true nature of the states of the problem. This approach deals with a stastical trajectory optimization procedure for unknown systems and extends it towards learning policies (optimal) which has less noise in them because of the lower variance in the optimal trajectories. 
* Contributions : 
    * Robust numerical implementation of EM which has been extensively used in system identification and extend it towards learning and generalization of optimal control for unseen initial conditions.
    * Efficient exploitation of the highly
uncertain explored state space, which is analytically quantified from the theoretical analysis of the covariance matrix.
    * Promising sample efficiency and good success of generalization from multiple testing instances

![example GIF](Door_opening.gif)


[Accepted Paper](https://arxiv.org/pdf/2010.00304.pdf)
[Video Link for Performance comparison](https://uoneduau-my.sharepoint.com/:v:/g/personal/c3297407_uon_edu_au/EWT4B72ad55HltG7VVKnmqQBivCYWAw_wO8szhIsXc4V7Q?e=C9U1cS)


