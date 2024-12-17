import os
from stable_baselines3 import PPO

class Agent:
    def __init__(self):
        # Navigate back two folders and into the "models" folder
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, '..', '..', 'models', '1v1Bot.zip', '1v1Bot')
        
        # Normalize the path to handle different operating systems
        model_path = os.path.normpath(model_path)

        # Load the model
        self.model = PPO.load(model_path)

    def act(self, state):
        # Evaluate your model here
        action, _ = self.model.predict(state, deterministic=True)
        return action
