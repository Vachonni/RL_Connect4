# Initialize agent
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

# Train agent
model.learn(total_timesteps=60000)


##

# Save agent
model.save("./models/first_agent")