import numpy as np
import os
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.state_setters import state_setter


class DuckyLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)
        
TICK_RATE = 120
TICK_SKIP = 8

def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
       FaceBallReward, EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from rlgym_sim.utils import common_values
    from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
    from util.rewards import SpeedTowardBallReward, InAirReward

    # Eviroment settings
    spawn_opponents = True # Bots will train agasint eachother
    team_size = 1 # Training 1v1s
    game_tick_rate = TICK_RATE
    tick_skip = TICK_SKIP # Iterate learning every 8 frames
    timeout_seconds = 10 # If ball isn't hit within 10 seconds, end the episode
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    # The LookupAction parser maps discrete action indices to predefined action vectors (throttle, steer, pitch, etc.),
    # enabling discrete RL policies to interact with the continuous action space required for Rocket League gameplay.
    action_parser = LookupAction()
    
    # End the episode if these conditions are met
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # Rewards that influence the bots behaviour. (reward, reward weight)
    reward_fn = CombinedReward.from_zipped(
	    (EventReward(touch=1), 50), # Giant reward for actually hitting the ball
	    (SpeedTowardBallReward(), 5), # Move towards the ball!
	    (FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
	    (InAirReward(), 0.15) # Make sure we don't forget how to jump
    )

    # Observation builder to convert state of game into data that bot trains with
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    # How begining of the episode is setup.
    # RandomState is being used to ensure concistancy in any position instead of always focusing on kickoffs
    state_setter = RandomState(ball_rand_speed=True,
                               cars_rand_speed=True,
                               cars_on_ground=False)

    # Create the enviroment using what has been setup above
    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = DuckyLogger()

    # Number of training processes to run
    n_proc = 50
    
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    
    # Whether or not to visualize bots training via 3D simulation
    render = True
    render_delay= TICK_SKIP / TICK_RATE
    
    # Target amount of steps to train before automatically ending
    timestep_limit = 1_000_000_000
    
    # Amount of steps in an interation
    ts_per_iteration = 50000
    
    # Amount of data that the learning algorithm trains on each iteration
    ppo_batch_size = ts_per_iteration 
    
    # Amount of data that the learning algorithm trains on each iteration
    ppo_minibatch_size = 25000
    
    # Data will be split into chunks of these size to conserve VRAM
    exp_buffer_size = ts_per_iteration * 3
    
    # How many layers the policy has. The more it has, the smarter and more complex it can be
    # There is a neural network for the policy the bot uses, and also for the critic that predicts the reward the policy will get
    policy_layer_sizes = [2048, 2048, 1024, 1024]
    critic_layer_sizes = policy_layer_sizes
    
    # How many times the learning phase is repeated on the same batch of data
    # Increasing will decrease steps per second, but will result in faster overall learning
    ppo_epochs = 2
    
    # This is the scale factor for entropy. Entropy fights against learning to make your bot pick actions more randomly
    # Higher value = exploration, lower value = exploitation
    ppo_ent_coef = 0.01
    
    # Neural network learning rate. Limits how much the policy can change to ensure learning continues in a consistant direction
    # Higher value = more change allowed, lower value = less change allowed
    policy_lr = 0.0002
    
    # wandb setup
    log_to_wandb = True 
    wandb_project_name = "rlgym-ppo-ducky"
    wandb_group_name = "ducky-ballchase"
    wandb_run_name = "ducky-ballchase-run"

    # Save/load path setup
    add_unix_timestamp = False;
    checkpoints_save_folder = os.path.join("data", "checkpoints", wandb_group_name, wandb_run_name)
    
    # Save every this number of timesteps
    save_every_ts = 5_000_000

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      render=render,
                      render_delay=render_delay,
                      timestep_limit=timestep_limit,
                      ts_per_iteration=ts_per_iteration,
                      ppo_batch_size=ppo_batch_size,
                      ppo_minibatch_size=ppo_minibatch_size,
                      exp_buffer_size=exp_buffer_size,
                      policy_layer_sizes=policy_layer_sizes,
                      critic_layer_sizes=critic_layer_sizes,
                      ppo_epochs=ppo_epochs,
                      ppo_ent_coef=ppo_ent_coef,
                      policy_lr=policy_lr,
                      log_to_wandb=log_to_wandb,
                      wandb_project_name=wandb_project_name,
                      wandb_group_name=wandb_group_name,
                      wandb_run_name=wandb_run_name,
                      add_unix_timestamp=add_unix_timestamp,
                      checkpoints_save_folder=checkpoints_save_folder,
                      save_every_ts=save_every_ts
                      )
    
    learner.learn()