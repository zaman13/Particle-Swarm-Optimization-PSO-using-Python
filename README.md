# Particle Swarm Optimization (PSO) using Python


<p float="left">
<a href = "https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python"> <img src="https://img.shields.io/badge/Language-Python-blue" alt="alt text"> </a>
<a href = "https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/LICENSE"> <img src="https://img.shields.io/github/license/zaman13/Particle-Swarm-Optimization-PSO-using-Python" alt="alt text"></a>
<a href = "https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/tree/master/Code"> <img src="https://img.shields.io/badge/version-1.2-red" alt="alt text"> </a>
</p>

Vectorized general particle swarm optimization code using python. 

The code can work with any arbitrary fitness/cost function with arbitrary number of optimization parameters (dimensions). To increase the processing speed, the code has been completely vectorized. All possible parallel operations are implemented using matrix mathematics. Thus, nested loops are avoided. Only a single for loop going over the iterations/generations is used.  


# Output animation
Generated using [pso_2D_animation_v1.py](https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/Code/pso_2D_animation_v1.py) file.


<p float="left">
<img src="https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/pso_anim_1.gif" alt="alt text" width="400" align = "top">
<img src="https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/pso_anim_2.gif" alt="alt text" width="400" align = "middle">
<img src="https://github.com/zaman13/Particle-Swarm-Optimization-PSO-using-Python/blob/master/colorbar.png" alt="alt text" height="240" align = "middle">

</p>

# Version Notes:

v1_1:
- Added average fitness and average pbest plots for checking convergence 

v1_2:
- The intial variable assignment has been embedded within the loop
- Added elapsed time calculator
