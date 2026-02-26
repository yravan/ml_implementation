"""
Gridworld Environments.

Implementation Status: STUB
Complexity: ★★☆☆☆ (Beginner-Intermediate)
Prerequisites: None

Simple gridworld environments for testing RL algorithms, including
navigation tasks, mazes, and goal-reaching problems.

References:
    - Sutton & Barto (2018): Reinforcement Learning: An Introduction
      http://incompleteideas.net/book/the-book.html
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class GridWorld:
    """
    Basic gridworld environment.

    A discrete grid where an agent navigates from start to goal,
    avoiding obstacles. Standard testbed for tabular RL.

    Theory:
        Gridworlds are finite MDPs where:
        - States: (row, col) positions
        - Actions: Up, Down, Left, Right
        - Transitions: Deterministic (with optional stochasticity)
        - Rewards: -1 per step, +1 at goal, penalty for obstacles

    Args:
        grid_size: (height, width) of the grid
        start: Starting position (row, col)
        goal: Goal position (row, col)
        obstacles: List of obstacle positions
        slip_prob: Probability of slipping to perpendicular action
        reward_goal: Reward for reaching goal
        reward_step: Reward per step
        reward_obstacle: Penalty for hitting obstacle

    Example:
        >>> env = GridWorld(grid_size=(5, 5), start=(0, 0), goal=(4, 4))
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(1)  # Move down
    """

    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (4, 4),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        slip_prob: float = 0.0,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_obstacle: float = -1.0,
        max_steps: int = 100
    ):
        """Initialize gridworld."""
        self.height, self.width = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.slip_prob = slip_prob
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.max_steps = max_steps

        # State
        self.agent_pos = None
        self.steps = 0

        # Spaces
        self.n_states = self.height * self.width
        self.n_actions = 4

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (agent position)
        """
        raise NotImplementedError(
            "Reset:\n"
            "- agent_pos = start\n"
            "- steps = 0\n"
            "- Return state representation"
        )

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Action index (0=up, 1=down, 2=left, 3=right)

        Returns:
            observation: New state
            reward: Reward received
            done: Whether episode ended
            info: Additional information
        """
        raise NotImplementedError(
            "Step:\n"
            "- Apply slip_prob for stochastic transitions\n"
            "- Compute next position based on action\n"
            "- Check boundaries and obstacles\n"
            "- Compute reward\n"
            "- Check if done (goal reached or max_steps)\n"
            "- Return (obs, reward, done, info)"
        )

    def _get_next_pos(
        self,
        pos: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """Compute next position given action."""
        raise NotImplementedError(
            "Get next position:\n"
            "- UP: (row-1, col)\n"
            "- DOWN: (row+1, col)\n"
            "- LEFT: (row, col-1)\n"
            "- RIGHT: (row, col+1)\n"
            "- Clip to boundaries"
        )

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to state index."""
        return pos[0] * self.width + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col)."""
        return (state // self.width, state % self.width)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get full transition probability matrix P[s, a, s'].

        Returns:
            Transition matrix [n_states, n_actions, n_states]
        """
        raise NotImplementedError(
            "Transition matrix:\n"
            "- For each state s and action a:\n"
            "  - Compute P(s'|s,a) considering slip_prob\n"
            "- Return P[s,a,s']"
        )

    def get_reward_matrix(self) -> np.ndarray:
        """
        Get reward matrix R[s, a].

        Returns:
            Reward matrix [n_states, n_actions]
        """
        raise NotImplementedError(
            "Reward matrix:\n"
            "- For each (s, a):\n"
            "  - Compute expected reward\n"
            "- Return R[s,a]"
        )

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: 'human' for console, 'rgb_array' for image

        Returns:
            RGB array if mode='rgb_array'
        """
        raise NotImplementedError(
            "Render gridworld:\n"
            "- Create grid visualization\n"
            "- Mark agent, goal, obstacles\n"
            "- Print or return image"
        )


class FourRooms(GridWorld):
    """
    Four Rooms environment.

    Classic hierarchical RL benchmark with four rooms connected
    by doorways. Tests temporal abstraction.

    References:
        - Sutton et al. (1999): Between MDPs and semi-MDPs
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (11, 11),
        **kwargs
    ):
        """Initialize Four Rooms."""
        # Define walls
        obstacles = self._create_walls(grid_size)
        super().__init__(
            grid_size=grid_size,
            obstacles=obstacles,
            **kwargs
        )

    def _create_walls(
        self,
        grid_size: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Create four rooms layout with doors."""
        raise NotImplementedError(
            "Create walls:\n"
            "- Horizontal wall at middle with 1 door\n"
            "- Vertical wall at middle with 1 door per room\n"
            "- Return list of wall positions"
        )


class CliffWalking(GridWorld):
    """
    Cliff Walking environment.

    Agent must navigate from start to goal along a cliff edge.
    Falling off the cliff gives large negative reward.

    Classic example showing difference between SARSA and Q-learning.
    """

    def __init__(
        self,
        width: int = 12,
        **kwargs
    ):
        """Initialize Cliff Walking."""
        # Cliff is bottom row except start and goal
        obstacles = [(3, c) for c in range(1, width - 1)]

        super().__init__(
            grid_size=(4, width),
            start=(3, 0),
            goal=(3, width - 1),
            obstacles=obstacles,
            reward_obstacle=-100.0,
            **kwargs
        )


class WindyGridworld(GridWorld):
    """
    Windy Gridworld environment.

    Standard gridworld with wind pushing the agent upward
    in certain columns.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (7, 10),
        wind: Optional[List[int]] = None,
        **kwargs
    ):
        """Initialize Windy Gridworld."""
        self.wind = wind or [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        super().__init__(
            grid_size=grid_size,
            start=(3, 0),
            goal=(3, 7),
            **kwargs
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step with wind effect."""
        raise NotImplementedError(
            "Step with wind:\n"
            "- Compute next pos from action\n"
            "- Apply wind: row -= wind[col]\n"
            "- Clip to grid\n"
            "- Return standard step outputs"
        )


class MazeEnv(GridWorld):
    """
    Maze environment.

    Randomly generated or predefined mazes for navigation.
    """

    def __init__(
        self,
        size: int = 10,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize maze."""
        obstacles = self._generate_maze(size, random_seed)

        super().__init__(
            grid_size=(size, size),
            obstacles=obstacles,
            **kwargs
        )

    def _generate_maze(
        self,
        size: int,
        seed: Optional[int]
    ) -> List[Tuple[int, int]]:
        """Generate random maze using DFS or Prim's algorithm."""
        raise NotImplementedError(
            "Generate maze:\n"
            "- Use DFS or Prim's algorithm\n"
            "- Ensure path exists from start to goal\n"
            "- Return list of wall positions"
        )
