# 1 Project Overview

Your goal is to design and implement the best possible DQN architecture for playing Pac-Man across multiple challenging layouts. Building on your homework implementation, you should experiment with advanced techniques, novel architectures, and sophisticated training strategies to create a robust agent that excels across diverse environments.

**Competition Element:** The team with the highest average performance will receive a single letter grade bonus. However, all teams will be graded on effort, not performance, to encourage experimentation.

# 2 Problem Statement

Create a DQN agent that achieves the highest possible average win rate across four distinct Pac-Man layouts:

- **classic**: Complex 19x21 maze with multiple ghosts and corridors
- **spiral**: Simple 7x7 spiral layout with predictable structure
- **spiral harder**: Same layout but with more aggressive ghost behavior
- **empty**: Open 7x7 space requiring different navigation strategies

**Challenge:**
Each layout presents unique strategic demands, requiring an agent that can generalize across diverse spatial structures and opponent behaviors.

# 3 Evaluation Methodology

Your final agent will be evaluated using a standardized protocol:

**Final Score:** Average win rate across all 200 games

$$
\text{Score} = \frac{1}{4} \left( \text{Win rate}_{classic} + \text{Win rate}_{spiral} + \text{Win rate}_{spiral\_harder} + \text{Win rate}_{empty} \right)
$$

Your final algorithm must work from image data alone. However, you can do whatever you want in the intermediate stages. Your code deliverables will typically include your `.pt` file along with your `dqn_agent.py`.

# 4 Suggested Research Directions

Explore any combination of the following advanced techniques:

## 4.1 Network Architecture Innovations

- **Dueling DQN**: Separate value and advantage streams
- **Multi-scale CNNs**: Different receptive fields for local vs. global features
- **Attention mechanisms**: Focus on relevant parts of the game state
- **Residual connections**: Enable deeper networks
- **Custom architectures**: Design novel network structures

## 4.2 Advanced DQN Algorithms

- **Prioritized Experience Replay**: Sample important transitions more frequently
- **Rainbow DQN**: Combine multiple DQN improvements
- **Distributional DQN**: Model full return distribution

## 4.3 Training Enhancements

- **Curriculum learning**: Progressive difficulty across layouts
- **Transfer learning**: Pre-train on simpler layouts
- **Multi-task learning**: Train on all layouts simultaneously
- **Hyperparameter optimization**: Systematic search for best settings
- **Data augmentation**: Rotation/reflection of game states

## 4.4 State Representation

- **Frame stacking**: Multiple consecutive frames as input
- **Feature engineering**: Hand-crafted spatial features
- **Auxiliary tasks**: Predict pellet locations, ghost movements
- **Multi-modal inputs**: Combine visual and symbolic information

# 5 Technical Report (4–6 pages)

- **Motivation and Related Work**: Survey of relevant DQN improvements
- **Methodology**: Detailed description of your approach and innovations
- **Experimental Setup**: Training procedures, hyperparameters, evaluation protocol
- **Results and Analysis**: Performance across layouts, ablation studies, failure cases
- **Discussion**: What worked, what didn’t, and why
- **Future Work**: Promising directions for further improvement

## 5.1 Presentation (20 minutes)

- Approach overview
- Key technical innovations
- Experimental results and insights
- Live demonstration of your best agent
- Q&A with class and instructor

**Note:** Grading emphasizes effort. I am expecting 20 hrs per group member.
