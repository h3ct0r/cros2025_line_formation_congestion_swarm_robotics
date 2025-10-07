from matplotlib.patches import Circle, Wedge
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import euclidean_distance
from robot import RobotStatus
import swarm_simulator
import yaml
import math


if __name__ == "__main__":
    param_file_path = "params/sim1.yaml"
    with open(param_file_path, "r") as file:
        sim_parameters = yaml.safe_load(file)

    # Define some static obstacles
    obstacles_list = []
    for idx in range(60, 120, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([50.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )
    for idx in range(-50, 42, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([50.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(40, 45, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([20.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(-10, 20, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([-60.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    for idx in range(-20, 10, 2):
        obstacles_list.append(
            swarm_simulator.Obstacle(
                np.array([10.0, idx]), influence_radius=sim_parameters["DELTA_J"]
            ),
        )

    # Initialize the simulation
    swarm_sim = swarm_simulator.SwarmSimulation(
        sim_parameters, obstacles=obstacles_list
    )

    print(
        f"Starting simulation with {sim_parameters["NUM_ROBOTS"]} robots goals {sim_parameters["GOALS_POS"]}."
    )
    print(f"Rearranging Regions: {sim_parameters["REARRANGING_REGIONS_RADII"]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()
    plt.show()

    # Run simulation steps and collect data
    num_simulation_steps = sim_parameters["NUM_SIMULATION_STEPS"]
    all_robot_data_history = []
    for step in range(num_simulation_steps):
        current_step_data = swarm_sim.run_simulation_step()
        all_robot_data_history.append(current_step_data)

        if step % 10 == 0 or step == num_simulation_steps - 1:
            leader_count = sum(
                1
                for _, _, _, state, _ in current_step_data
                if state == RobotStatus.LEADER
            )
            in_line_count = sum(
                1
                for _, _, _, state, _ in current_step_data
                if state == RobotStatus.IN_LINE
            )
            group_count = sum(
                1
                for _, _, _, state, _ in current_step_data
                if state == RobotStatus.GROUP
            )
            print(
                f"Step {step}: Leader={leader_count}, InLine={in_line_count}, Group={group_count}"
            )

            # Check if all robots have reached a small vicinity around the final target
            robots_at_goal = [
                1 for r in swarm_sim.robots if r.state == RobotStatus.FINISHED
            ]
            if len(robots_at_goal) == sim_parameters["NUM_ROBOTS"]:
                print(f"All robots reached the goal at step {step}!")
                break

            # This part is outside the core algorithm implementation but helps visualize results.
            # Plot the final state of the swarm
            final_positions_states = all_robot_data_history[-1]
            ax.cla()

            # Plot robots based on their state
            robot_dict = {r.id: r for r in swarm_sim.robots}
            for id, pos, orientation, state, front_pos in final_positions_states:
                color = "blue"  # 'group' state (blue in Fig. 4)
                if state == RobotStatus.IN_LINE:
                    color = "green"  # 'inLine' state (green in Fig. 4)
                elif state == RobotStatus.LEADER:
                    color = "red"  # 'leader' state (red in Fig. 4)
                elif state == RobotStatus.FINISHED:
                    color = "black"

                # robot heading
                ax.plot(
                    [pos[0], front_pos[0]],
                    [pos[1], front_pos[1]],
                    "-",
                    color="black",
                    linewidth=2,
                )
                robot_radius = Circle(
                    pos,
                    sim_parameters["ROBOT_MARKER_RADIUS"],
                    color=color,
                    fill=True,
                    alpha=0.6,
                    linestyle=":",
                )
                ax.add_patch(robot_radius)

                # plt.text(
                #     pos[0] - 0.8,
                #     pos[1] - 0.8,
                #     f"{id} {state.name}",
                #     size=8,
                #     color="black",
                # )
                # flocking, ignoring the back of the robot
                # a1 = (
                #     math.degrees(orientation)
                #     - sim_parameters["FLOCKING_ANGLE_DEGREES"] / 2.0
                # ) % 360.0
                # a2 = (
                #     math.degrees(orientation)
                #     + sim_parameters["FLOCKING_ANGLE_DEGREES"] / 2.0
                # ) % 360.0
                # wedge_flocking = Wedge(
                #     pos,
                #     sim_parameters["FLOCKING_RADIUS"],
                #     a1,
                #     a2,
                #     color="b",
                #     alpha=0.05,
                # )
                # ax.add_patch(wedge_flocking)

                # # separation radius, smaller than the flocking radius
                # sep_radius = Circle(
                #     pos,
                #     sim_parameters["SEPARATION_RADIUS"],
                #     color="blue",
                #     fill=False,
                #     linestyle=":",
                #     alpha=0.2,
                # )
                # ax.add_patch(sep_radius)

                following_robot = robot_dict[id].inline_following_robot
                if following_robot is not None:
                    pos_following = following_robot.position
                    ax.plot(
                        [pos[0], pos_following[0]],
                        [pos[1], pos_following[1]],
                        color="red",
                    )

            # Plot target position
            for idx, goal_pos in enumerate(sim_parameters["GOALS_POS"]):
                ax.plot(
                    [goal_pos[0]],
                    [goal_pos[1]],
                    "x",
                    color="purple",
                    markersize=10,
                    label="Goal" if idx == 0 else None,
                )

                rearranging_regions_colors = ["orange", "pink", "red"]
                # Plot rearranging regions
                for idx2, R_radius in enumerate(
                    sim_parameters["REARRANGING_REGIONS_RADII"]
                ):
                    # Plotting only the first one with a label to avoid multiple legend entries for "Rearranging Region"
                    rearrange_circle = Circle(
                        goal_pos,
                        R_radius,
                        color=rearranging_regions_colors[idx2],
                        fill=False,
                        linestyle="--",
                        label=(
                            (f"Rearranging Region (R={R_radius})") if idx == 0 else None
                        ),
                    )
                    ax.add_patch(rearrange_circle)

            # Plot obstacles
            for obs in obstacles_list:
                ax.plot(
                    [obs.position[0]],
                    [obs.position[1]],
                    "s",
                    color="black",
                    markersize=10,
                    label="Obstacle" if obs == obstacles_list else "",
                )
                obs_influence_circle = Circle(
                    obs.position.tolist(),
                    obs.influence_radius,
                    color="gray",
                    alpha=0.2,
                )
                ax.add_patch(obs_influence_circle)

            ax.set_title(f"Final Swarm State (Step {swarm_sim.time_step_count})")
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(sim_parameters["PLOT_X_RANGE"])
            ax.set_ylim(sim_parameters["PLOT_Y_RANGE"])
            ax.grid(True)
            ax.legend()

            plt.draw()
            plt.pause(0.01)

    print("\nSimulation finished.")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final plot
