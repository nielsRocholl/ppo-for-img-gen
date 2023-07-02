# Image Reconstruction with Deep RL

![Example Image Reconstruction](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDVpcnl6YXc5MW1tMGt6b3FkMWkzODhucGdmNmhlZmE4bXNlNmQxNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/N7LXuxyfeVfttrzYvA/giphy.gif)

This project explores the application of deep Reinforcement Learning (RL) algorithms in image reconstruction, utilizing the popular PPO algorithm.

The task involves selecting a simple, binarized image (akin to pixel art or images of MNIST digits), which the agent must reconstruct within an environment represented by a grid, each cell signifying a pixel. The goal is to maximize the similarity between the recreated and the target image.

## The Image Environment

The environment, denoted as `E`, is an `N x N` grid, with each cell (`C`) representing a pixel. The cells within this environment can be in one of two states, `C=1` or `C=0`.

```math
E = [e_{i,j} ∈ {0, 1} | 1 ≤ i, j ≤ N]
````


The agent (`A`) operating within this environment has an action space `A = {0, 1}`, corresponding to changing a cell's state to 0 (black pixel) or 1 (white pixel). The agent initiates its trajectory at the top-left cell (`e_{1,1}`) and terminates at the bottom-right cell (`e_{N,N}`).

```math
P = {e_{i_1,j_1}, e_{i_2,j_2}, ..., e_{i_k,j_k}}, where 1 ≤ k ≤ N², i_1 = j_1 = 1 and i_k = j_k = N
```


Here `P` is the path of the agent, representing the sequence of cells it visits, altering their states with actions from `A`.

The agent's observation at any time step `t`, denoted as `O`, is a tuple consisting of the current environment state `$E_{s_t}`, the agent's current position `$P_{s_t}`, and the target image `T`.

```math
O_t = ({E_{s_{t-i}}}{i=0}^{f-1}, {P{s{t-i}}}_{i=0}^{f-1}, T)
```


In this definition, `fps` denotes the frame-per-second parameter. If `fps = f`, the agent's observation at time `t` includes the states of the environment and the agent's positions for the `f-1` previous time steps.

## Reward Signal

The reward signal was designed to encourage an equilibrium of potential rewards for both black and white cells, facilitating the accurate replication of the target image. The total possible reward `R` is split equally between white cells `$C_w` and black cells `$C_b`.

For an agent at position `$e_{i, j}`, corresponding to target image location `$t_{i, j}`, the reward is calculated as follows:

- For `$t_{i, j}=1`, `$r=R_{w}/C_{w}` if `$e_{i, j}=t_{i, j}`, otherwise `$r=-(R_{w}/C_{w})`;
- For `$t_{i, j}=0`, `$r=R_{b}/C_{b}` if `$e_{i, j}=t_{i, j}`, otherwise `$r=-(R_{b}/C_{b})`.

If the final step is reached with `S=T`, an extra reward of `0.1R` is given.
