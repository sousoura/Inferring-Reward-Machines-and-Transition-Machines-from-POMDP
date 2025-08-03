from typing import List
import numpy as np  # 添加这行
from agent import Agent
from grid_environment import GridEnvironment
from grid_state import GridState


class DemoRunner:
    """Runs demonstration of learned policy"""

    def __init__(self, agent: Agent, environment: GridEnvironment):
        self.agent = agent
        self.env = environment

    def run_demo(self, max_steps: int = 20, verbose: bool = True):
        """Run a single demonstration episode"""
        state = self.env.reset()
        total_reward = 0.0
        trajectory = []

        if verbose:
            print("\nDemo Run:")
            print("Initial state:")
            self.env.render()

        for step in range(max_steps):
            # Choose action (no exploration)
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            action = self.agent.choose_action(state)
            self.agent.epsilon = old_epsilon

            # Take action
            result = self.env.step(action)
            total_reward += result.reward

            # Record trajectory
            trajectory.append({
                'step': step,
                'state': state,
                'action': action,
                'reward': result.reward,
                'next_state': result.state,
                'done': result.done,
                'info': result.info
            })

            if verbose:
                action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                print(f"\nStep {step + 1}, Action: {action_names[action]}, Reward: {result.reward:.3f}")
                if result.info.get('label') is not None:
                    print(f"Triggered label: {result.info['label']}")
                self.env.render()

            if result.done:
                if verbose:
                    print(f"\nReached terminal state! Total reward: {total_reward:.3f}")
                break

            state = result.state

        return total_reward, trajectory

    def run_multiple_demos(self, num_demos: int = 10, verbose: bool = False):
        """Run multiple demonstration episodes"""
        demo_rewards = []

        for i in range(num_demos):
            reward, _ = self.run_demo(max_steps=1000000, verbose=verbose)
            demo_rewards.append(reward)
            print(f"Demo {i + 1}: Reward = {reward:.3f}")

        print(f"\nAverage demo reward: {np.mean(demo_rewards):.3f}")
        return demo_rewards
