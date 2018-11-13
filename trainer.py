import numpy as np
from collections import deque
import torch

def trainer(agent, env, brain_name, 
            n_episodes=2000, max_t=1000, score_solved=13.0,
            save_model=True, model_filename='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        agent: the agent
        env: the environment
        brain_name: unity environment brain_name
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        score_solved (float): score (averaged on the last 100 episodes) at which we consider the environment solved
        save_model (bool): if we save the model weights or not
        model_filename (str): path for saving the model weights
    """

    scores = []                        
    scores_window = deque(maxlen=100)                      
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0]            
        score = 0
        for t in range(max_t):
            # Choose action
            action = agent.act(state)
            
            # Send action to env, get state and reward
            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations[0]   
            reward = env_info.rewards[0]                   
            done = env_info.local_done[0]
            
            # Update the agent
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
        
        scores_window.append(score)       
        scores.append(score)              
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=score_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if save_model:
                torch.save(agent.actor_local.state_dict(), 'actor_ ' + model_filename)
                torch.save(agent.critic_local.state_dict(), 'critic_ ' + model_filename)
            break
    
    return scores, i_episode