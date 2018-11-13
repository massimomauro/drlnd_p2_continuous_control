This repository contains the solution for the second project of the Udacity Deep Reinforcement Learning Nanodegree.

# The Environment

The environment for this project is the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, two separate versions of the Unity environment were provided:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

**I solved the environment using the first version with just a single agent**. **According to project requirements, the agent must get an average score of +30 over 100 consecutive episodes to be considered solved.**

# How to

Install the dependancies first

* **Option 1: pipenv (recommended).** Initialize a pipenv environment by running `pipenv --three install` inside the root directory of the repo. [Pipenv](http://docs.pipenv.org/) will automatically locate the [Pipfiles](https://github.com/pypa/pipfile), create a new virtual environment and install the necessary packages.

* **Option 2: pip.** Install the needed dependencies by running `pip install -r requirements.txt` 

A solution of the environment can be obtained by running the `DDPG_Continuous_Control.ipynb` notebook.

## Repository structure

*  `DDPG_Continuous_Control.ipynb` notebook contains a solution of the environment with single agent
*  `DDPG_Continuous_Control-Multi.ipynb` notebook contains an adaptation of the previous environment to the multi-agent case, but no solution was produced in that case (you should be able to obtain one by running it)
*  `Report.md` contains a description of the implementation.
*  `trainer.py`/`trainer_multi.py` contain the code for running the training of the agent over a given number of episodes in the two environments
*  `ddpg_agent.py` contains the implementation of the DDPG agent
*  `model.py` contains the definition of the Actor and Critic networks
*  `actor_solution.pth` and `critic_solution.pth` contain the model weights of actor and critic network after environment was solved





