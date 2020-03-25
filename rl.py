EPISODES_list = [100,200,300,400,500]
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
episode_reward_list = deque(maxlen=100)

for EPISODES in EPISODES_list:
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(1000):
          action = agent.act(state)
          next_state, reward, done, _ = env.step(action)
          total_reward += reward
          next_state = np.reshape(next_state, [1, state_size])
          agent.remember(state, action, reward, next_state, done)
          state = next_state
          if done:
              break
          if len(agent.memory) > batch_size:
              agent.replay(batch_size)
        episode_reward_list.append(total_reward)
    episode_reward_list = list(episode_reward_list)
    episode_reward_avg = sum(episode_reward_list[len(episode_reward_list)-100:])/100
    # print("episode: {}/{}, score: {}, e: {:.2}, last 100 episodes rew: {:.2f}".format(e, EPISODES, total_reward, agent.epsilon, episode_reward_avg))
    print(f"Episode: {EPISODES} with average reward: {episode_reward_avg}")