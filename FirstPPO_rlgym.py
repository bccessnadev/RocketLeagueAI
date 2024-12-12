try:
	import rlgym

	from stable_baselines3 import PPO

	from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward

	# My Parameters
	model_path = "models/firstPPO.zip"
	training_total_timesteps = 1_000_000

	#Make the default rlgym environment with ball chase reward
	env = rlgym.make(reward_fn=LiuDistancePlayerToBallReward())

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
		model.learn(training_total_timesteps)
		model.save(model_path)
	except KeyboardInterrupt:
		print("Exiting training and saving model")
		model.save(model_path)
except Exception as e:
	print("An error has occured:", e)

input("Press Enter to close...")