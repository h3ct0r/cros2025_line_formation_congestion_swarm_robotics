from enum import Enum
import numpy as np
import math
from helper_functions import euclidean_distance, normalize_vector, normalize_angle_rad


class RobotStatus(Enum):
    GROUP = 1
    IN_LINE = 2
    LEADER = 3
    FINISHED = 4


class Robot:
    """Represents a single robot in the swarm."""

    def __init__(self, robot_id, position, velocity, sim_params):
        self.sim_params = sim_params

        self.id = robot_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orientation = normalize_angle_rad(0)
        self.radius = sim_params["ROBOT_MARKER_RADIUS"]
        self.front_position = self.position + np.array(
            (
                self.radius * np.cos(self.orientation),
                self.radius * np.sin(self.orientation),
            )
        )
        self.state = RobotStatus.GROUP  # Initial state for all robots [4]
        self.target_index = 0
        self.inline_following_robot = None

        self.previous_position = self.position

        # Stores previous distance to detect crossing rearranging regions [10]
        curr_goal_pos = np.array(
            sim_params["GOALS_POS"][self.target_index], dtype=float
        )

        # Parameters are stored locally for clarity, though global constants are used
        self.katt = sim_params["K_ATT"]
        self.katt_inline = sim_params["K_ATT_INLINE"]
        self.krep = sim_params["K_REP"]
        self.delta_j = sim_params["DELTA_J"]
        self.kcoh = sim_params["K_COH"]
        self.kali = sim_params["K_ALI"]
        self.ksep = sim_params["K_SEP"]
        self.flocking_radius = sim_params["FLOCKING_RADIUS"]
        self.flocking_angle_degrees = sim_params["FLOCKING_ANGLE_DEGREES"]
        self.separation_radius = sim_params["SEPARATION_RADIUS"]
        self.vmax = sim_params["V_MAX"]
        self.vmin = sim_params["V_MIN"]
        self.rearranging_regions_radii = sim_params["REARRANGING_REGIONS_RADII"]

    @staticmethod
    def iterative_inline_target_definition(robot_to_follow, all_robots):
        """
        Iteratively assigns robots to follow each other in a line formation.

        Given a starting robot (`robot_to_follow`) and a list of all robots (`all_robots`), this method:
        1. Finds the closest neighbor(s) within the flocking radius of the current robot.
        2. Sets the closest neighbor's state to "inLine" and assigns its `inline_following_robot` to the current robot's object.
        3. Repeats the process for the next closest robot, using the remaining neighbors.

        Args:
            robot_to_follow (Robot): The robot that others should follow in the line.
            all_robots (list[Robot]): List of all robots to consider for line formation.

        Returns:
            None

        Note:
            - The iteration terminates when there are no more neighbors to assign.
            - This method modifies the state of robots in-place.
        """
        remaining_robots = all_robots.copy()
        current_robot = robot_to_follow

        while remaining_robots:
            flocking_neighbors = Robot.get_sorted_neighbors_in_radius(
                current_robot,
                remaining_robots,
                current_robot.flocking_radius * 2,
                limit_angle_degrees=360,
            )

            if not flocking_neighbors:
                break

            closest_robot = flocking_neighbors[0]
            # if closest_robot.id == current_robot.id:
            #     print("--" * 60)
            #     print(
            #         "\n\n\n\n\n\n\n\n\n\n\n\n\n\nSAME ID!!!!!!!!!!!!\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
            #     )
            #     print("--" * 60)

            closest_robot.state = RobotStatus.IN_LINE
            closest_robot.inline_following_robot = current_robot

            # Remove the assigned robot from the remaining list
            remaining_robots.remove(closest_robot)

            # Move to the next robot in the line
            current_robot = closest_robot

    @staticmethod
    def get_sorted_neighbors_in_radius(
        target_robot, all_robots, radius, limit_angle_degrees=270
    ):
        """
        Identifies neighboring robots within a specified radius and angle.
        The angle filters neighbors based on the robot's current heading (velocity direction).
        Returns a list of neighbors sorted by distance to the target robot
        """
        neighbors = []
        for other_robot in all_robots:
            if other_robot.id == target_robot.id:
                continue

            dist = euclidean_distance(target_robot.position, other_robot.position)
            if dist <= radius:
                if limit_angle_degrees < 360:
                    # Calculate angle relative to current robot's direction
                    angle_rad = target_robot.get_angle_to_robot(other_robot)
                    a1 = (
                        target_robot.orientation
                        - math.radians(limit_angle_degrees / 2.0)
                    ) % math.radians(360)
                    a1 = normalize_angle_rad(a1)

                    a2 = (
                        target_robot.orientation
                        + math.radians(limit_angle_degrees / 2.0)
                    ) % math.radians(360)
                    a2 = normalize_angle_rad(a2)

                    if angle_rad < math.radians(360 - limit_angle_degrees):
                        # Check if within the angular sector
                        neighbors.append(other_robot)
                else:
                    neighbors.append(other_robot)  # All-around neighborhood

        neighbors.sort(
            key=lambda r: euclidean_distance(r.position, target_robot.position)
        )
        return neighbors

    def _calculate_attractive_force(self, goal_pos):
        """Calculates the attractive force towards a given goal position (Eq 5)."""
        if self.state in [RobotStatus.LEADER, RobotStatus.IN_LINE]:
            return self.katt_inline * (goal_pos - self.position)
        else:
            return self.katt * (goal_pos - self.position)

    def _calculate_repulsive_force(self, obstacle_pos, influence_dist):
        """Calculates the repulsive force from an obstacle or neighbor (Eq 6)."""
        d = euclidean_distance(self.position, obstacle_pos)
        # Avoid division by zero when rho is very small
        if d < influence_dist and d > 1e-6:
            return (
                self.krep
                * ((1 / d) - (1 / influence_dist))
                * ((self.position - obstacle_pos) / d)
            )
        else:
            return np.array([0.0, 0.0])

    def _calculate_cohesion_force(self, flocking_neighbors):
        """Calculates the cohesion force (Fcoh), directing robot towards neighbors' center of mass (Eq 7)."""
        if not flocking_neighbors:
            return np.array([0.0, 0.0])

        center_of_mass = sum(n.position for n in flocking_neighbors) / len(
            flocking_neighbors
        )
        return self.kcoh * (center_of_mass - self.position)

    def _calculate_alignment_force(self, flocking_neighbors):
        """Calculates the alignment force (Fali), matching robot's velocity to neighbors' average (Eq 8)."""
        if not flocking_neighbors:
            return np.array([0.0, 0.0])

        avg_velocity = sum(n.velocity for n in flocking_neighbors) / len(
            flocking_neighbors
        )
        return self.kali * (avg_velocity - self.velocity)

    def _calculate_separation_force(self, separation_neighbors):
        """Calculates the separation force (Fsep), preventing collisions with close neighbors (Eq 9)."""
        if not separation_neighbors:
            return np.array([0.0, 0.0])

        avg_neighbor_pos = sum(n.position for n in separation_neighbors) / len(
            separation_neighbors
        )
        # The formula in [17, Eq 9] is Fsep(qi) = ksep * (qi - SUM(qj)/|Mi|), implying repulsion from center of nearby flockmates.
        return self.ksep * (self.position - avg_neighbor_pos)

    def get_angle_to_robot(self, robot_target):
        """
        Calculates the absolute angular difference between this robot's orientation and the direction to another robot.

        Args:
            robot2: An object representing the other robot, expected to have a 'position' attribute.

        Returns:
            float: The absolute value of the angle (in radians) between this robot's current orientation and the vector pointing from this robot to robot2.
        """
        vet = robot_target.position - self.position
        return np.abs(self.orientation - np.arctan2(vet[1], vet[0]))

    def compute_potential_field_force(self, target_for_pf, all_robots, obstacles):
        """
        Computes the total potential field force (Fpf) for the robot (Eq 4).
        'target_for_pf' can be the main goal or a neighbor's position depending on the robot's state.
        """
        Fpf = self._calculate_attractive_force(target_for_pf)  # -∇Uatt(qi) [2]

        # Consider other robots as dynamic obstacles for repulsion [5]
        neighbors_for_repulsion = Robot.get_sorted_neighbors_in_radius(
            self, all_robots, self.delta_j
        )
        for other_robot in neighbors_for_repulsion:
            Fpf += self._calculate_repulsive_force(
                other_robot.position, self.delta_j
            )  # -∇Urep,j(qi) [2]

        # Consider static obstacles
        for obstacle in obstacles:
            Fpf += self._calculate_repulsive_force(
                obstacle.position, obstacle.influence_radius
            )
        return Fpf

    def compute_flocking_force(self, all_robots):
        """Computes the total flocking force (Ff) by combining cohesion, alignment, and separation (Eq 10)."""
        flocking_neighbors = Robot.get_sorted_neighbors_in_radius(
            self, all_robots, self.flocking_radius, self.flocking_angle_degrees
        )
        # Separation uses a smaller radius [9]
        separation_neighbors = Robot.get_sorted_neighbors_in_radius(
            self, all_robots, self.separation_radius
        )

        Fcoh = self._calculate_cohesion_force(flocking_neighbors)
        Fali = self._calculate_alignment_force(flocking_neighbors)
        Fsep = self._calculate_separation_force(separation_neighbors)

        return Fcoh + Fali + Fsep

    def update_state(self, all_robots):
        """
        Updates the robot's state based on the finite state machine (Fig. 3)
        and line formation strategy [4, 9, 10].
        """
        curr_goal_pos = np.array(
            self.sim_params["GOALS_POS"][self.target_index], dtype=float
        )
        dist_to_goal = euclidean_distance(self.position, curr_goal_pos)
        if any(
            dist_to_goal < radius
            for radius in self.sim_params["REARRANGING_REGIONS_RADII"]
        ):
            robots_w_same_goal = [
                r for r in all_robots if r.target_index == self.target_index
            ]
            for r in robots_w_same_goal:
                r.state = RobotStatus.GROUP
                self.inline_following_robot = None

        if dist_to_goal <= self.sim_params["REACH_GOAL_RADIUS"]:
            if self.target_index < len(self.sim_params["GOALS_POS"]) - 1:
                self.target_index += 1
            else:
                print(f"r:{self.id} is now FINISHED")
                self.state = RobotStatus.FINISHED

        if self.state == RobotStatus.FINISHED or self.state != RobotStatus.GROUP:
            return

        robots_w_same_goal = [
            r for r in all_robots if r.target_index == self.target_index
        ]
        curr_goal_pos = np.array(
            self.sim_params["GOALS_POS"][self.target_index], dtype=float
        )
        closest_robot_to_goal = min(
            robots_w_same_goal,
            key=lambda r: euclidean_distance(r.position, curr_goal_pos),
        )
        dist_to_target = euclidean_distance(
            closest_robot_to_goal.position, curr_goal_pos
        )
        # Check if closest robot is inside the congestion zone (R0)
        if dist_to_target < self.rearranging_regions_radii[0]:
            closest_robot_to_goal.state = RobotStatus.LEADER
            all_robots_no_leader = [
                r for r in robots_w_same_goal if r.id != closest_robot_to_goal.id
            ]
            Robot.iterative_inline_target_definition(
                closest_robot_to_goal, all_robots_no_leader
            )

    def calculate_resultant_velocity(self, all_robots, obstacles):
        """
        Calculates the resultant velocity for the robot based on its current state (Eq 11, 12).
        """
        if self.state == RobotStatus.FINISHED:
            return

        curr_goal_pos = np.array(
            self.sim_params["GOALS_POS"][self.target_index], dtype=float
        )

        alpha = 1  # Default for 'group' state (combines PF and Flocking) [4]
        target_for_pf = curr_goal_pos  # Default attractive target for PF

        if self.state == RobotStatus.LEADER:
            alpha = 0  # No flocking for leader [11]
            target_for_pf = curr_goal_pos  # Leader moves directly to the target [11]
        elif self.state == RobotStatus.IN_LINE:
            alpha = 0  # No flocking for inLine robots [11]
            following_robot = None
            for r in all_robots:
                if r == self.inline_following_robot:
                    following_robot = r
                    break

            if following_robot:
                target_for_pf = following_robot.position
            else:
                self.inline_following_robot = None

        # Compute forces based on the determined state and parameters
        Fpf = self.compute_potential_field_force(target_for_pf, all_robots, obstacles)
        Ff = self.compute_flocking_force(all_robots)
        Fc = Fpf + (alpha * Ff)  # Combined force (Eq 11)

        # Apply the control law to limit velocity (Eq 12)
        Fc_norm = np.linalg.norm(Fc)
        if Fc_norm == 0:
            self.velocity = np.array([0.0, 0.0])  # No force, no movement
        else:
            # Clamp speed between vmin and vmax
            desired_speed = np.clip(Fc_norm, self.vmin, self.vmax)

            # Set direction and scaled speed
            self.velocity = desired_speed * normalize_vector(Fc)

    def update_kinematics(self, dt):
        """Updates the robot's position based on its current velocity and the time step."""

        if self.state == RobotStatus.FINISHED:
            return

        self.previous_position = self.position.copy()

        # update position
        vel_delta = self.velocity * dt
        self.position += vel_delta

        # update orientation
        self.orientation = normalize_angle_rad(np.arctan2(vel_delta[1], vel_delta[0]))
        self.front_position = self.position + np.array(
            (
                self.radius * np.cos(self.orientation),
                self.radius * np.sin(self.orientation),
            )
        )

    def __repr__(self):
        return f"{self.id}:({self.state.name}->{self.inline_following_robot.id if self.inline_following_robot is not None else None}) goal:{self.target_index}"
