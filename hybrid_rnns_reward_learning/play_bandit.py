# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Interactive four-armed drifting bandit task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import signal
import sys
from typing import Sequence


@dataclass(frozen=True)
class GameConfig:
  """Configuration for a bandit block."""

  n_arms: int = 4
  n_trials: int = 150
  response_deadline_s: float | None = None
  decay: float = 0.9836
  drift_std: float = 2.8
  observation_std: float = 4.0
  center: float = 50.0
  reward_min: float = 1.0
  reward_max: float = 100.0


def _clip_reward(value: float, config: GameConfig) -> float:
  return min(config.reward_max, max(config.reward_min, value))


def generate_block(config: GameConfig, rng: random.Random) -> list[list[float]]:
  """Generates one block of drifting rewards."""

  latent_means = [
      [0.0 for _ in range(config.n_arms)] for _ in range(config.n_trials)
  ]
  payouts = [[0.0 for _ in range(config.n_arms)] for _ in range(config.n_trials)]

  latent_means[0] = [
      float(rng.randint(int(config.reward_min), int(config.reward_max)))
      for _ in range(config.n_arms)
  ]
  payouts[0] = [
      _clip_reward(
          latent_means[0][arm] + rng.gauss(0.0, config.observation_std), config
      )
      for arm in range(config.n_arms)
  ]

  for trial in range(1, config.n_trials):
    for arm in range(config.n_arms):
      previous_mean = latent_means[trial - 1][arm]
      current_mean = (
          config.decay * previous_mean
          + (1.0 - config.decay) * config.center
          + rng.gauss(0.0, config.drift_std)
      )
      current_mean = _clip_reward(current_mean, config)
      latent_means[trial][arm] = current_mean
      payouts[trial][arm] = _clip_reward(
          current_mean + rng.gauss(0.0, config.observation_std), config
      )

  return payouts


def _readline_with_timeout(prompt: str, timeout_s: float | None) -> str | None:
  """Reads one line from stdin, optionally aborting on timeout."""

  if timeout_s is None:
    try:
      return input(prompt)
    except EOFError:
      return None

  if not hasattr(signal, "setitimer"):
    return _readline_with_timeout(prompt, None)

  def _raise_timeout(unused_signum, unused_frame):
    raise TimeoutError

  previous = signal.signal(signal.SIGALRM, _raise_timeout)
  try:
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    return input(prompt)
  except TimeoutError:
    print("\nTime limit exceeded.")
    return None
  except EOFError:
    return None
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous)


def get_choice(
    config: GameConfig,
    trial_index: int,
    auto_play: bool,
    rng: random.Random,
) -> int | None:
  """Gets a player choice or returns None for a missed trial."""

  if auto_play:
    return rng.randrange(config.n_arms)

  prompt = f"Trial {trial_index + 1:>3}/{config.n_trials} | Choose 1-4 (or q): "
  raw_choice = _readline_with_timeout(prompt, config.response_deadline_s)
  if raw_choice is None:
    return None

  text = raw_choice.strip().lower()
  if text == "q":
    raise KeyboardInterrupt
  if text in {"1", "2", "3", "4"}:
    return int(text) - 1

  print("Invalid response. Trial counted as missed.")
  return None


def _format_arm_labels(n_arms: int) -> str:
  return "  ".join(f"[{arm + 1}]" for arm in range(n_arms))


def _feedback_line(choice: int | None, reward: float) -> str:
  if choice is None:
    return "Missed trial. Reward: 0.0"
  return f"You chose arm {choice + 1}. Reward: {reward:.1f}"


def _show_instructions(config: GameConfig, auto_play: bool) -> None:
  print("Four-Armed Drifting Bandit")
  print("==========================")
  print(
      "Choose one of four arms on each trial. Rewards drift over time, so the"
      " best option can change."
  )
  print("You only see the reward from the arm you chose.")
  if config.response_deadline_s is None:
    print("No response deadline is enforced in this local version.")
  else:
    print(
        f"Response deadline: {config.response_deadline_s:.1f} s."
        " Missed trials give 0 reward."
    )
  if auto_play:
    print("Auto-play mode is enabled for this run.")
  print()
  print("Arms:", _format_arm_labels(config.n_arms))
  print()


def summarize_block(
    payouts: Sequence[Sequence[float]],
    choices: Sequence[int | None],
    rewards: Sequence[float],
) -> None:
  """Prints end-of-block diagnostics."""

  total_reward = sum(rewards)
  optimal_total = sum(max(trial_payouts) for trial_payouts in payouts)
  random_total = sum(sum(trial_payouts) / len(trial_payouts) for trial_payouts in payouts)
  misses = sum(choice is None for choice in choices)

  print()
  print("Block complete")
  print("==============")
  print(f"Your total reward: {total_reward:.1f}")
  print(f"Theoretical upper bound for this block: {optimal_total:.1f}")
  print(f"Expected reward from random choice: {random_total:.1f}")
  print(f"Missed trials: {misses}")
  print("Choice counts:")
  for arm in range(len(payouts[0])):
    count = sum(choice == arm for choice in choices)
    print(f"  Arm {arm + 1}: {count}")


def play_block(config: GameConfig, seed: int, auto_play: bool) -> None:
  """Runs one block of the bandit task."""

  rng = random.Random(seed)
  payouts = generate_block(config, rng)
  _show_instructions(config, auto_play)

  choices: list[int | None] = []
  rewards: list[float] = []

  try:
    for trial_index in range(config.n_trials):
      choice = get_choice(config, trial_index, auto_play, rng)
      reward = 0.0 if choice is None else float(payouts[trial_index][choice])
      choices.append(choice)
      rewards.append(reward)

      print(_feedback_line(choice, reward))
      print(f"Cumulative reward: {sum(rewards):.1f}")
      print()
  except KeyboardInterrupt:
    print("\nEnding block early.")

  if choices:
    summarize_block(payouts[: len(choices)], choices, rewards)
  else:
    print("No trials completed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Play a four-armed drifting bandit task in the terminal."
  )
  parser.add_argument(
      "--trials",
      type=int,
      default=150,
      help="Number of trials in the block.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for the generated block.",
  )
  parser.add_argument(
      "--deadline",
      type=float,
      default=None,
      help="Optional response deadline in seconds. Use 4 to mimic the lab task.",
  )
  parser.add_argument(
      "--auto-play",
      action="store_true",
      help="Play automatically with random actions. Useful for smoke tests.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  config = GameConfig(
      n_trials=args.trials,
      response_deadline_s=args.deadline,
  )
  play_block(config, seed=args.seed, auto_play=args.auto_play)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
