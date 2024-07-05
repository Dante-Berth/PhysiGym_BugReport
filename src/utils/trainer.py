def Trainer(env, agent, parameters):
    replay_buffer = ReplayBuffer(max_size=parameters["max_size"], device=agent.device)
    batch_size = parameters["batch_size"]
    num_steps = parameters["num_steps"]
    state = env.reset()
    state = state[0]
    total_reward = 0
    for i in range(num_steps):
        action = agent.action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

        if len(replay_buffer.buffer) >= batch_size:
            agent.train(replay_buffer, batch_size)

        if done:
            state = env.reset()
            state = state[0]
            total_reward = 0
        if i % 100 == 0:
            print("Step: {}, Total Reward: {:.2f}".format(i, total_reward))

    env.close()
