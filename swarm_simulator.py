import numpy as np
from helper_functions import euclidean_distance
from robot import Robot


class Obstacle:
    """Represents a static obstacle in the environment."""

    def __init__(self, position, influence_radius=1.0):
        self.position = np.array(position, dtype=float)
        self.influence_radius = influence_radius


class SwarmSimulation:
    """Manages the overall swarm simulation."""

    def __init__(self, sim_params, obstacles=None):
        self.sim_params = sim_params
        self.num_robots = sim_params["NUM_ROBOTS"]
        self.target_pos = np.array(
            sim_params["GOALS_POS"][0], dtype=float
        )  # first goal
        self.initial_robot_pos = np.array(sim_params["INITIAL_ROBOT_POS"], dtype=float)
        self.arena_size = sim_params["ARENA_SIZE"]
        self.obstacles = obstacles if obstacles is not None else []
        self.robots = self._initialize_robots()
        self.time_step_count = 0

    def _initialize_robots(self):
        """
        Initializes robots in a circular region outside the congestion zone [12].
        For simplicity, this implementation places them randomly within a square,
        ensuring they are beyond a certain radius from the target.
        """
        robots = []
        # Define the initial spawning area relative to the target, outside the congestion zone
        # We ensure they start further out than R0 * 1.5 to provide space for initial flocking.
        spawn_radius = self.sim_params["REARRANGING_REGIONS_RADII"][0] * 1.5

        initial_positions = []
        while len(initial_positions) < self.num_robots:
            # Generate random positions within a square around the target
            # This is a simplification; a true circular distribution would use polar coordinates.
            x = np.random.uniform(
                self.initial_robot_pos[0] - (self.arena_size / 8),
                self.initial_robot_pos[0] + (self.arena_size / 8),
            )
            y = np.random.uniform(
                self.initial_robot_pos[1] - (self.arena_size / 8),
                self.initial_robot_pos[1] + (self.arena_size / 8),
            )
            pos = np.array([x, y])

            # Ensure robots start outside the initial spawn_radius from the target
            if euclidean_distance(pos, self.target_pos) > spawn_radius:
                initial_positions.append(pos)

        for i in range(self.num_robots):
            position = initial_positions[i]
            # Assign random initial velocities
            velocity = np.array(
                [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)]
            )
            robots.append(Robot(i, position, velocity, self.sim_params))
        return robots

    def run_simulation_step(self):
        """Executes one step of the swarm simulation."""
        self.time_step_count += 1

        # 1. Update robot states
        # States are updated first, as they determine how forces are calculated.
        # The update_state method contains logic for leader election, in-line joining (ξ=true),
        # and state resets due to rearranging regions.
        for r in self.robots:
            r.update_state(self.robots)

        # 2. Calculate new velocities based on (potentially new) states
        # This must happen after all states are updated for the current step.
        for robot in self.robots:
            robot.calculate_resultant_velocity(self.robots, self.obstacles)

        # 3. Update positions based on the new velocities
        for robot in self.robots:
            robot.update_kinematics(self.sim_params["DT"])

        for r in self.robots:
            print(
                f"[run_simulation_step] r:{r}"  # {Robot.get_sorted_neighbors_in_radius(r, self.robots, self.sim_params["FLOCKING_RADIUS"])
            )
        print("-" * 60 + "\n")

        # Return current positions and states for logging or visualization
        return [
            (r.id, r.position, r.orientation, r.state, r.front_position)
            for r in self.robots
        ]
