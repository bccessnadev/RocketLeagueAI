try:
	import rlgym

	from stable_baselines3 import PPO
	from stable_baselines3.common.callbacks import CheckpointCallback
	from stable_baselines3.common.vec_env import VecMonitor

	from rlgym.envs import Match

	from rlgym.utils.obs_builders import AdvancedObs
	from rlgym.utils.state_setters import DefaultState
	from rlgym.utils.action_parsers import DiscreteAction
	from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
	from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
	from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
	from rlgym.utils.reward_functions import CombinedReward

	from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition

	from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

	if __name__ == '__main__':  # Required for multiprocessing
		# My Parameters
		model_folder = "models/1v1Bot.zip"
		model_name = "1v1Bot"
		model_path = model_folder + "/" + model_name + ".zip"
		num_instances = 1
		training_total_timesteps = 50_000_000
		frame_skip = 8 # Number of ticks to repeat an action
		fps = 120 / frame_skip
		max_episode_time=300
		max_no_touch_time = 45
		save_frequency = 5_000_000
		team_size = 1

		rewards = [
			EventReward(goal=100, concede=-100, shot=10, save=30, demo=20), 
			VelocityPlayerToBallReward(), 
			VelocityBallToGoalReward(),
		]

		reward_weights = [1.0, 0.25, 0.5]

		# Define function used to create an env for each instance
		def get_match():
			return Match(
				team_size=team_size,
				tick_skip=frame_skip,
				reward_function=CombinedReward(rewards, reward_weights),
				spawn_opponents=True,
				terminal_conditions=[TimeoutCondition(fps * max_episode_time), NoTouchTimeoutCondition(fps * max_no_touch_time), GoalScoredCondition()],
				obs_builder=AdvancedObs(),  # Not that advanced, good default
				state_setter=DefaultState(),  # Resets to kickoff position
				action_parser=DiscreteAction()  # Discrete > Continuous (less training time). A button is either pressed or it isn't
			)
	
		env = SB3MultipleInstanceEnv(get_match, num_instances)
		env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard

		# Save model every so often
		# Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
		# This saves to specified folder with a specified name
		callback = CheckpointCallback(round(save_frequency / env.num_envs), save_path=model_folder, name_prefix=model_name)

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

		input("Press Enter to close...")

except Exception as e:
	print("An error has occured:", e)
	input("Press Enter to close...")
