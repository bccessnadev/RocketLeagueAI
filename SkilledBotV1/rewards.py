from rlgym.utils.reward_functions import RewardFunction

class InAirReward(RewardFunction): # We extend the class "RewardFunction"

        from rlgym.utils.gamestates import GameState, PlayerData
        
        # Empty default constructor (required)
        def __init__(self):
                super().__init__()

        # Called when the game resets (i.e. after a goal is scored)
        def reset(self, initial_state: GameState):
                pass # Don't do anything when the game resets

        # Get the reward for a specific player, at the current state
        def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
                # "player" is the current player we are getting the reward of
                # "state" is the current state of the game (ball, all players, etc.)
                # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
                if not player.on_ground:
                    # We are in the air! Return full reward
                    return 1
                else:
                    # We are on ground, don't give any reward
                    return 0



