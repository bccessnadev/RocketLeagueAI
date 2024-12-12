try:
	import rlgym

	from stable_baselines3 import PPO
	from stable_baselines3.common.callbacks import CheckpointCallback

	from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
	from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
	from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
	from rlgym.utils.reward_functions import CombinedReward

	# My Parameters
	model_folder = "models/BasicRewards.zip"
	model_name = "BasicRewards"
	model_path = model_folder + "/" + model_name + ".zip"
	training_total_timesteps = 10_000_000
	save_frequency = 1_000_000

	rewards = [
		EventReward(goal=100, concede=-100, shot=10, save=30, demo=20), 
		VelocityPlayerToBallReward(), 
		VelocityBallToGoalReward(),
	]

	reward_weights = [1.0, 0.5, 1.0]
	
	#Make the default rlgym environment with ball chase reward
	env = rlgym.make(
		reward_fn=CombinedReward(rewards, reward_weights)
	)

	# Save model every so often
	callback = CheckpointCallback(round(save_frequency), save_path=model_folder, name_prefix=model_name)

	try:
		model = PPO.load(model_path, env)
		print("Loaded previous save")
	except:
		print("No model found. Creating new one")
	
		#Initialize PPO from stable_baselines3
		model = PPO("MlpPolicy", env=env, verbose=1)

	# Allow for ending via keyboard interupt
	try:	
		#Train our agent!
		model.learn(training_total_timesteps, callback=callback)
		model.save(model_path)
	except KeyboardInterrupt:
		print("Exiting training and saving model")
		model.save(model_path)
except Exception as e:
	print("An error has occured:", e)

input("Press Enter to close...")