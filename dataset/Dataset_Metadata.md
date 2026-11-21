# Dataset Overview

This dataset contains agent-level observations collected from simulated aerial swarm scenarios under varying swarm sizes, obstacle configurations, and attrition conditions.

Each row corresponds to an individual agent at a particular simulation timestep and contains information about the simulation scenario, environment, agent kinematics, swarm geometry, neighboring agents, graph-based connectivity, and agent criticality.

The dataset is intended for analyzing and predicting the criticality of agents within an aerial swarm under different environmental and attrition conditions.

# Variable Descriptions

## 1. Scenario and Environment Variables

### `case_id`
- **Description:** Unique identifier for each simulation scenario.
- **Example:** `c001_n60_none` indicates case #1, a swarm with 60 agents, and no attrition.

### `num_agents`
- **Description:** Total number of agents in the swarm (critical + decoy) at the beginning of the scenario.

### `n_obstacles`
- **Description:** Number of obstacles present in the environment.
- **Details:** Obstacles can be represented as spheres, cylinders, or boxes. The number of obstacles varies across scenarios.

### `obstacle_density`
- **Description:** Fraction of the 200 × 200 simulation area covered by obstacles.
- **Range:** 0 to 1.
- **Interpretation:**
  - `0` represents no obstacle coverage.
  - Larger values indicate greater obstacle coverage.

### `mean_obstacle_size`
- **Description:** Average radius or half-width of the obstacles present in the environment.

### `clutter_index`
- **Description:** A measure of how difficult the environment is to navigate.
- **Details:** It combines information about obstacle density and obstacle count/size so that environments containing either a few large obstacles or many smaller obstacles can be represented as highly cluttered.
- **Interpretation:**
  - A value near `0` represents an open environment with little or no obstruction.
  - Higher values indicate increasingly cluttered or obstructed environments.

### `obstacle_clustering`
- **Description:** Describes how spatially concentrated the obstacles are.
- **Interpretation:**
  - `0` indicates that obstacles are distributed relatively independently throughout the environment.
  - `1` indicates that obstacles are strongly concentrated into a small number of regions, such as clusters of buildings in an urban environment.

## 2. Simulation and Agent Identification

### `step`
- **Description:** Simulation timestep at which the observation was recorded.
- **Details:** The dataset is sampled every second simulation step (`sample_every = 2`).

### `agent_id`
- **Description:** Unique index identifying an individual agent within a given simulation scenario.

## 3. Agent Position and Distance Variables

### `rel_x`, `rel_y`, `rel_z`
- **Description:** Agent position relative to the swarm centroid along the x, y, and z axes.

### `dist_to_centroid`
- **Description:** Euclidean distance between the agent and the swarm centroid.

### `rank_dist_to_centroid`
- **Description:** Percentile rank of the agent's distance to the swarm centroid among the agents alive at that timestep.
- **Interpretation:**
  - `0` indicates the closest agent.
  - `1` indicates the farthest agent.

### `dist_to_target`
- **Description:** Euclidean distance between the agent and the mission target point.

### `rank_dist_to_target`
- **Description:** Percentile rank of the agent's distance to the target among the agents at that timestep.

## 4. Agent Kinematic Variables

### `vx`, `vy`, `vz`
- **Description:** Velocity components of the agent along the x, y, and z axes.

### `speed`
- **Description:** Scalar speed of the agent according to the Dubins-model state.

### `heading`
- **Description:** Agent yaw angle, describing its horizontal direction of travel.

### `flight_path_angle`
- **Description:** Agent flight-path angle (pitch/climb angle), describing its vertical direction of travel.

### `accel`
- **Description:** Along-track acceleration command applied to the agent during the previous simulation step.

### `heading_rate`
- **Description:** Yaw-rate command applied to the agent during the previous simulation step.

## 5. Neighbor and Local Swarm Geometry Variables

### `num_neighbors`
- **Description:** Number of other agents located within the agent's sensing range.
- **Sensing range:** 55 units.

### `mean_neighbor_dist`
- **Description:** Mean Euclidean distance between the agent and its neighboring agents within the sensing range.

### `min_neighbor_dist`
- **Description:** Distance from the agent to its closest neighboring agent.

### `weighted_density`
- **Description:** Distance-weighted measure of local neighbor density.
- **Details:** Nearby agents contribute more strongly than agents farther away.

### `angular_coverage`
- **Description:** Fraction of angular sectors surrounding the agent that contain neighboring agents.
- **Details:** The angular space is divided into 12 azimuth sectors and 4 elevation bins.
- **Purpose:** This feature measures how well-surrounded or spatially escorted an agent is by other swarm members.

## 6. Criticality Variables

### `criticality_score`
- **Description:** A continuous score between 0 and 1 representing the estimated criticality of an agent based on the geometry and state of the swarm.
- **Purpose:** It provides a continuous measure of how critical or high-value an agent appears to be.
- **Important distinction:** This score is distinct from `is_critical`, which is the ground-truth binary label.

### `is_critical`
- **Description:** Ground-truth binary label indicating whether the agent is an actual critical (high-value) asset or a decoy.
- **Values:**
  - `1` = critical agent
  - `0` = decoy agent

## 7. Graph-Based Centrality Variables

The following variables are standard graph-centrality measures computed from the instantaneous neighbor graph of the swarm using NetworkX. They describe the structural position and importance of an agent within the swarm's connectivity graph at that timestep.

### `degree_centrality`
- **Description:** Measures the relative connectivity of an agent based on the number of neighboring agents it is directly connected to.

### `closeness_centrality`
- **Description:** Measures how close an agent is, in terms of graph distance, to the other agents in the swarm connectivity graph.

### `betweenness_centrality`
- **Description:** Measures how frequently an agent lies on the shortest paths between other agents in the swarm graph, indicating its potential role as a connectivity bridge.

### `eigenvector_centrality`
- **Description:** Measures an agent's connectivity while also considering the importance/connectivity of the agents to which it is connected.

## 8. Attrition Conditions

The dataset contains different levels of decoy-agent attrition.

### `None`
- **Description:** No drones are lost during the simulation.

### `Light`
- **Description:** 1–2 decoy drones are lost, with losses occurring over 1–2 timesteps.

### `Medium`
- **Description:** 2–3 decoy drones are lost, with losses occurring over 2–3 timesteps.

### `Heavy`
- **Description:** 3–4 decoy drones are lost, with losses occurring over 3–4 timesteps.

### Effect of Attrition

Attrition affects the number of agents present in the swarm over time. Features such as neighbor counts, distances, density, angular coverage, and graph centrality are therefore computed based on the agents present/alive at each timestep.