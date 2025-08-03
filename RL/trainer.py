import numpy as np
from typing import List, Tuple
from agent import Agent, QRM_Agent
from grid_environment import GridEnvironment
from state import BaseState, State
from grid_state import GridState


class Trainer:
    """Manages training process"""

    def __init__(self, agent: Agent, environment):
        self.agent = agent
        self.env = environment

    def train_episode(self, max_steps: int = 1000000) -> Tuple[float, int]:
        """Train for one episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            # Agent chooses action
            action = self.agent.choose_action(state)

            # Environment executes action
            result = self.env.step(action)

            # Agent learns from experience
            self.agent.update(state, action, result.reward,
                              result.state, result.done)

            total_reward += result.reward
            steps += 1

            if result.done:
                break

            state = result.state

        return total_reward, steps

    def train(self, num_episodes: int = 1000) -> List[float]:
        """Train for multiple episodes"""
        rewards = []

        for episode in range(num_episodes):
            reward, steps = self.train_episode()
            rewards.append(reward)

            # Decay exploration
            self.agent.decay_epsilon()

            # Progress report
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward:.3f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")

        return rewards

class Mask_trainer:
    def __init__(self, agent: Agent, environment, mask_env):
        self.agent = agent
        self.env = environment
        self.mask_env = mask_env

    def train_episode(self, max_steps: int = 1000000) -> Tuple[float, int]:
        """Train for one episode"""
        self.env.reset()
        self.mask_env.reset()
        tm, rm = self.mask_env.get_tm_rm()

        label = str(self.env.get_label_now())

        tm.transition(label)
        rm.transition(label)

        old_tm_state, old_rm_state = tm.current_state, rm.current_state
        old_position = self.env.get_current_position()
        old_state = GridState(old_position[0], old_position[1], old_tm_state, old_rm_state)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            # Agent chooses action
            action = self.agent.choose_action(old_state)

            old_position = self.env.get_current_position()
            old_tm_state, old_rm_state = tm.current_state, rm.current_state

            # Environment executes action
            result = self.env.step(action)
            position = self.env.get_current_position()

            label = str(result.info["label"])

            tm.transition(label)
            rm.transition(label)

            tm_state, rm_state = tm.current_state, rm.current_state

            state = GridState(position[0], position[1], tm_state, rm_state)
            old_state = GridState(old_position[0], old_position[1], old_tm_state, old_rm_state)

            # Agent learns from experience
            self.agent.update(old_state, action, result.reward,
                              state, result.done)

            total_reward += result.reward
            steps += 1

            if result.done:
                # print("done")
                break

            old_state = state

            # print(label, result.state.trans_state, tm_state, result.state.reward_state, rm_state)

        # input()
        return total_reward, steps

    def train(self, num_episodes: int = 1000) -> List[float]:
        """Train for multiple episodes"""
        rewards = []

        for episode in range(num_episodes):
            reward, steps = self.train_episode()
            rewards.append(reward)

            # Decay exploration
            self.agent.decay_epsilon()

            # Progress report
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward:.3f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")

        return rewards


class Mask_QRM_trainer:
    def __init__(self, qrm_agent: QRM_Agent, environment):
        self.qrm_agent = qrm_agent
        self.env = environment

    def train_episode(self, max_steps: int = 1000000) -> Tuple[float, int]:
        """Train for one episode"""
        self.env.reset()

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            old_position = self.env.get_current_position()
            old_trans_state = self.env.get_current_trans_state()
            old_state = GridState(old_position[0], old_position[1], old_trans_state, "None")

            # Agent chooses action
            action = self.qrm_agent.choose_action(old_state)

            # Environment executes action
            result = self.env.step(action)

            position = self.env.get_current_position()
            trans_state = self.env.get_current_trans_state()
            state = GridState(position[0], position[1], trans_state, "None")

            label = str(result.info["label"])

            if self.qrm_agent.rm.output(
                        old_state.row + old_state.col * 26,
                        action,
                        old_trans_state.replace("_", "")):
                self.qrm_agent.update(old_state, action, result.reward, state, result.done, label)
            else:
                self.qrm_agent.update_only(old_state, action, result.reward, state, result.done, label)

            total_reward += result.reward
            steps += 1

            if result.done:
                # print("done")
                break

            # print(label, result.state.reward_state, self.qrm_agent.get_current_rm_state())

        # input()
        return total_reward, steps

    def train(self, num_episodes: int = 1000) -> List[float]:
        """Train for multiple episodes"""
        rewards = []

        for episode in range(num_episodes):
            reward, steps = self.train_episode()
            rewards.append(reward)

            # Decay exploration
            self.qrm_agent.decay_epsilon()
            self.qrm_agent.reset()

            # Progress report
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print(f"Episode {episode}, Average Reward: {avg_reward:.3f}, "
                      f"Epsilon: {self.qrm_agent.epsilon:.3f}")

        return rewards
