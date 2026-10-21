"""
Single decision-maker ("single authority") for the Committee vs. Authority
study.

The agent faces a binary decision about a hidden state `theta` in {-mu, +mu}.
It observes a noisy, possibly biased signal

    s = theta + b + sigma * z,    z ~ N(0, 1)

and chooses the action sign(s). With no bias, the probability of choosing the
correct action is

    p = Phi(mu / sigma)

so an agent can be parameterized either by its noise level `sigma` or directly
by its accuracy `p` (sigma = mu / Phi^-1(p)). This links the Condorcet view of
an agent (accuracy `p`) to the agent-based view (noisy Gaussian signal).

A single authority decides alone, so its decision time is T = 1 and its
quality-per-unit-time equals its accuracy.

The standard-normal draw `z` can be passed in from outside, so a committee can
later inject correlated errors across its members (shared bias, information
cascades) without changing this class.

Usage:
    > python agent.py --p 0.9
    > python agent.py --sigma 0.5 --bias 0.2 --num_trials 200000
"""

import argparse
import dataclasses
import statistics
from typing import Any, Dict, Optional, Union

import numpy as np

_STD_NORMAL = statistics.NormalDist()

ArrayLike = Union[float, np.ndarray]


# #############################################################################
# Agent
# #############################################################################


@dataclasses.dataclass
class Agent:
    """
    Single decision-maker with a noisy, possibly biased Gaussian signal.

    :param sigma: standard deviation of the signal noise (lower is more
        expert)
    :param bias: systematic error `b` added to every signal
    :param mu: magnitude of the hidden state, i.e., theta in {-mu, +mu}
    :param decision_time: time cost T of reaching a decision
    :param name: label used in reports
    """

    sigma: float
    bias: float = 0.0
    mu: float = 1.0
    decision_time: float = 1.0
    name: str = "authority"

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma={self.sigma} must be > 0")
        if self.mu <= 0:
            raise ValueError(f"mu={self.mu} must be > 0")
        if self.decision_time <= 0:
            raise ValueError(
                f"decision_time={self.decision_time} must be > 0"
            )

    @classmethod
    def from_accuracy(
        cls, p: float, *, mu: float = 1.0, **kwargs: Any
    ) -> "Agent":
        """
        Build an agent whose unbiased accuracy is `p`.

        :param p: probability of choosing the correct action when unbiased
        :param mu: magnitude of the hidden state
        :param kwargs: other `Agent` fields (e.g., `bias`, `name`)
        :return: agent with `sigma = mu / Phi^-1(p)`
        """
        if not 0.5 < p < 1.0:
            raise ValueError(f"p={p} must be in (0.5, 1)")
        sigma = mu / _STD_NORMAL.inv_cdf(p)
        return cls(sigma=sigma, mu=mu, **kwargs)

    @property
    def accuracy(self) -> float:
        """
        Return the analytical probability of choosing the correct action.

        Assumes an equal prior on theta = +mu and theta = -mu. A bias helps
        on one state and hurts on the other, so it always lowers accuracy.
        """
        p_pos = _STD_NORMAL.cdf((self.mu + self.bias) / self.sigma)
        p_neg = _STD_NORMAL.cdf((self.mu - self.bias) / self.sigma)
        return 0.5 * (p_pos + p_neg)

    @property
    def quality_per_unit_time(self) -> float:
        """
        Return accuracy / T, the metric used to compare with committees.
        """
        return self.accuracy / self.decision_time

    def observe(
        self,
        theta: ArrayLike,
        *,
        rng: Optional[np.random.Generator] = None,
        z: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """
        Draw the agent's signal about the hidden state.

        :param theta: hidden state(s), each in {-mu, +mu}
        :param rng: random generator used when `z` is not given
        :param z: standard-normal noise draw(s); pass these in to control
            the error correlation across agents
        :return: signal(s) `theta + bias + sigma * z`
        """
        theta = np.asarray(theta, dtype=float)
        if z is None:
            rng = rng if rng is not None else np.random.default_rng()
            z = rng.standard_normal(theta.shape)
        z = np.asarray(z, dtype=float)
        return theta + self.bias + self.sigma * z

    def log_odds(self, signal: ArrayLike) -> np.ndarray:
        """
        Return the agent's log-odds that theta = +mu given its signal.

        The agent does not know its own bias, so this is the likelihood
        ratio of an unbiased Gaussian observer: 2 * mu * s / sigma^2. Summing
        log-odds across agents gives Bayesian pooling weighted by competence.
        """
        signal = np.asarray(signal, dtype=float)
        return 2.0 * self.mu * signal / self.sigma**2

    def belief(self, signal: ArrayLike) -> np.ndarray:
        """
        Return the agent's posterior probability that theta = +mu.
        """
        return 1.0 / (1.0 + np.exp(-self.log_odds(signal)))

    def decide(self, signal: ArrayLike) -> np.ndarray:
        """
        Map signal(s) to action(s) in {-1, +1}.

        Ties (s == 0) have probability zero and are broken toward +1.
        """
        signal = np.asarray(signal, dtype=float)
        return np.where(signal >= 0, 1, -1)

    def act(
        self,
        theta: ArrayLike,
        *,
        rng: Optional[np.random.Generator] = None,
        z: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """
        Observe the hidden state and decide in one step.

        :return: action(s) in {-1, +1}
        """
        return self.decide(self.observe(theta, rng=rng, z=z))


# #############################################################################
# Simulation
# #############################################################################


def sample_states(
    num_trials: int, mu: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw hidden states uniformly from {-mu, +mu}.
    """
    return rng.choice([-mu, mu], size=num_trials)


def simulate(
    agent: Agent, num_trials: int, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Run `num_trials` independent decisions and summarize performance.

    :param agent: decision-maker to evaluate
    :param num_trials: number of independent decision problems
    :param seed: seed for reproducibility
    :return: empirical and analytical accuracy, standard error, decision
        time, and quality-per-unit-time
    """
    if num_trials <= 0:
        raise ValueError(f"num_trials={num_trials} must be > 0")
    rng = np.random.default_rng(seed)
    theta = sample_states(num_trials, agent.mu, rng)
    actions = agent.act(theta, rng=rng)
    correct = actions == np.sign(theta)
    empirical_accuracy = float(correct.mean())
    stderr = float(
        np.sqrt(empirical_accuracy * (1.0 - empirical_accuracy) / num_trials)
    )
    return {
        "sigma": agent.sigma,
        "bias": agent.bias,
        "empirical_accuracy": empirical_accuracy,
        "stderr": stderr,
        "analytical_accuracy": agent.accuracy,
        "decision_time": agent.decision_time,
        "quality_per_unit_time": empirical_accuracy / agent.decision_time,
    }


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--p", type=float, help="Unbiased accuracy of the agent (0.5, 1)"
    )
    group.add_argument("--sigma", type=float, help="Signal noise std dev")
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--decision_time", type=float, default=1.0)
    parser.add_argument("--num_trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    kwargs = dict(
        bias=args.bias, mu=args.mu, decision_time=args.decision_time
    )
    if args.sigma is not None:
        agent = Agent(sigma=args.sigma, **kwargs)
    else:
        # Default to the solo expert from the "Key Examples" section.
        p = args.p if args.p is not None else 0.9
        agent = Agent.from_accuracy(p, **kwargs)
    results = simulate(agent, args.num_trials, seed=args.seed)
    print(f"Agent: {agent}")
    for key, value in results.items():
        print(f"  {key:<22} {value:.4f}")


if __name__ == "__main__":
    _main(_parse())
