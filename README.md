# cros2025_line_formation_congestion_swarm_robotics
This repository is a python implementation of the CROS 2025 paper intitled: `Towards a Line Formation Algorithm to Reduce Congestion in Swarm Robotics` (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11066136)

```
This work proposes an algorithm for line formation in swarm robotics, designed to reduce congestion in environments with spatial constraints. The methodology combines a flocking algorithm to maintain group cohesion with a line formation strategy, enabling robots to organize sequentially when traversing narrow areas.We conducted a series of simulations, varying the number of robots, to evaluate the proposed approach. The results demonstrate that the proposed approach significantly reduces congestion and the time required for a group of robots to pass through a common point. By combining flocking and line formation strategies, we present a promising solution for efficient swarm navigation in environments with spatial constraints, optimizing coordinated movement, and minimizing delays.
```

The code was implemented in `numpy` with visualization using `matplotlib`.

<div align="center">
  <img src="https://github.com/user-attachments/assets/720b8419-bad1-4aab-901e-0c3397b42b6b" style="text-align:center; width:70%">
</div>


# How to run

- Edit the configuration file:
  ```
  vim params/sim1.yaml
  ```

- Run the program:
  ```
  python main.py
  ```
