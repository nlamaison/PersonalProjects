import dataclasses
import enum
import collections
import logging
from typing import Callable, Type, TypeVar

from abc import ABC, abstractmethod

T = TypeVar('T', bound='PlannerBase')

import numpy as np
from fastquadtree import QuadTree 

from simulation.generate_sim_world import SimWorld 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class GraphNode:
    """Graph node.
    
    q: configuration
    cost: cost associated with the configuration relative to the goal
    idx: the node's index in the graph
    prev_idx: index of previous node in the graph 
    """
    q: np.ndarray
    cost: float | np.floating 
    idx: int
    prev_idx: int | None = None

@dataclasses.dataclass
class PlanningResult:
    """Planning result.
    
    path: sequence of nodes denoting the valid path from init to goal configuration
    graph: collection of all sampled nodes
    """
    path: list[GraphNode] | None
    graph: list[GraphNode]


class PlanningTerminationCondition(enum.Enum):
    """Determines when to terminate the graph search."""
    MAX_ITERATIONS_REACHED = 0
    GOAL_CONFIGURATION_FOUND = 1

class PlannerBase(ABC):
    """Sampling-based planner implementation."""
    # Registry for factory create method. 
    _registry: dict[str, Type["PlannerBase"]] = {}
    _zero_tolerance: float = 1e-6

    class InvalidPlannerError(Exception):
        """Raised when an invalid planner type is provided when creating a planner object."""
        pass

    def __init__(self, convergence_epsilon: float, max_iterations: int) -> None:
        self._convergence_epsilon: float = convergence_epsilon
        self._max_iterations: int = max_iterations
        super().__init__()

    @classmethod
    def register_planner(cls, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a planner subclass with a string key."""
        def decorator(_subclass: Type[T]) -> Type[T]:
            if not issubclass(_subclass, PlannerBase):
                raise TypeError(f"Registered planner {_subclass.__name__} must be a subclass of PlannerBase.")

            if key in cls._registry:
                LOGGER.warning(f"Planner key '{key}' is being overwritten. Previously registered: {cls._registry[key]}")

            cls._registry[key] = _subclass
            return _subclass

        return decorator

    @classmethod
    def get_planner_class(cls, key: str) -> Type["PlannerBase"]:
        """Get the planner class for a given key."""
        try:
            return cls._registry[key]
        except KeyError:
            error_msg = f"Could not find planner class for key: '{key}'. Available keys: {list(cls._registry.keys())}"
            raise cls.InvalidPlannerError(error_msg) from KeyError

    @classmethod
    def create_planner(cls, key: str, *args, **kwargs) -> "PlannerBase":
        """Factory method to create a planner instance."""
        planner_class = cls.get_planner_class(key)
        return planner_class(*args, **kwargs)

    @abstractmethod
    def plan(self, workspace: SimWorld, init_q: np.ndarray, goal_q: np.ndarray) -> PlanningResult:
        raise NotImplementedError("Base class method should not be called!")

    @property
    def zero_tolerance(self) -> float:
        return self._zero_tolerance

    @property
    def convergence_epsilon(self) -> float:
        return self._convergence_epsilon

    @property
    def max_iterations(self) -> int:
        return self._max_iterations


@PlannerBase.register_planner('rrt')
class RRTPlanner(PlannerBase):
    """Implements the RRT sampling-based planning algorithm."""

    def __init__(
        self, 
        step_size: float, 
        convergence_epsilon: float, 
        max_iterations: int,
    ) -> None:
        self._step_size: float = step_size
        super().__init__(convergence_epsilon, max_iterations)

    def _sample_point(self, bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> np.ndarray:
        return np.array([np.random.uniform(bounds_x[0], bounds_x[1]), np.random.uniform(bounds_y[0], bounds_y[1])])

    def plan(self, workspace: SimWorld, init_q: np.ndarray, goal_q: np.ndarray, termination_condition: PlanningTerminationCondition = PlanningTerminationCondition.GOAL_CONFIGURATION_FOUND) -> PlanningResult:
        """Generate a graph.
        
        Graph is represented as a list[GraphNode]. Search will terminate when a sampled 
        node is within `convergence_epsilon` of the goal configuration. 
        """
        # Initialize graph.
        graph = [GraphNode(init_q, np.linalg.norm(goal_q - init_q), 0, None)]

        # Use KD-tree to do nearest neighbor search.
        # TODO(nico): Parameterize capacity for the quadtree? Not sure how to set it
        nns_tree = QuadTree(bounds=(workspace.bounds_x[0], workspace.bounds_y[0], workspace.bounds_x[1], workspace.bounds_y[1]), capacity=20, track_objects=True)
        nns_tree.insert((init_q[0], init_q[1]), obj=graph[0])

        # Validate initial configuration is collision-free.
        if workspace.check_collision(init_q):
            error_msg = f"Initial configuration {init_q} is in collision!"
            raise ValueError(error_msg)

        i = 0
        while (i <= self.max_iterations):
            # NOTE: Always increment the iteration counter, even if we have to bail out early.
            # TODO(nico): If we find we're often terminating search early, we can implement a separate
            # termination condition/add a separate iteration counter to break out of planning early 
            # if an error persists (e.g. distance to rand_q is too small). That way, we actually sample 
            # self._step_size nodes.
            i += 1

            rand_q = self._sample_point(workspace.bounds_x, workspace.bounds_y)
            near_q_node = nns_tree.nearest_neighbor((rand_q[0], rand_q[1]), as_item=True)
            if near_q_node is None or near_q_node.obj is None:
                warning_msg = f"Could not find nearest neighbor to sample point {rand_q}. Will try sampling another point."
                LOGGER.warning(warning_msg)
                continue

            # Generate new configuration.
            near_q: np.ndarray = near_q_node.obj.q
            near_q_idx: int = near_q_node.obj.idx

            direction_vec = rand_q - near_q
            distance_to_rand = float(np.linalg.norm(direction_vec))
            
            # Skip if rand_q is too close to near_q.
            # NOTE: Don't count this as a failed iteration.
            if distance_to_rand <= self.zero_tolerance:
                continue

            direction_u = direction_vec / distance_to_rand
            # Clamp step size.
            step_size = min(self.step_size, distance_to_rand)
            new_q = near_q + step_size * direction_u

            # Skip if new_q is outside workspace bounds.
            # NOTE: Don't count this as a failed iteration.
            if (new_q[0] < workspace.bounds_x[0] or new_q[0] > workspace.bounds_x[1] or
                new_q[1] < workspace.bounds_y[0] or new_q[1] > workspace.bounds_y[1]):
                continue

            # Skip if new_q is in collision.
            # NOTE: Don't count this as a failed iteration.
            if workspace.check_collision(new_q):
                continue

            new_q_cost = np.linalg.norm(goal_q - new_q)
            new_node = GraphNode(new_q, new_q_cost, len(graph), near_q_idx)
            graph.append(new_node)
            nns_tree.insert((new_q[0], new_q[1]), obj=new_node)

            # Check distance to goal.
            # NOTE: In `MAX_ITERATIONS_REACHED` mode, we do not return a path - just the graph.
            if new_q_cost <= self.convergence_epsilon and termination_condition == PlanningTerminationCondition.GOAL_CONFIGURATION_FOUND:
                # Trace back valid path and return planning result.
                path: collections.deque[GraphNode] = collections.deque()
                path.appendleft(new_node)
                current_node = new_node
                while current_node.prev_idx != None:
                    current_node = graph[current_node.prev_idx]
                    path.appendleft(current_node)

                return PlanningResult(path=list(path), graph=graph)

        if termination_condition == PlanningTerminationCondition.GOAL_CONFIGURATION_FOUND:
            warning_msg = f"Failed to find a valid path within {self.max_iterations} iterations!"
            LOGGER.warning(warning_msg)

        return PlanningResult(path=None, graph=graph)

    @property
    def step_size(self) -> float:
        return self._step_size


PLANNER_SUBCLASS_TYPES = [t for t in PlannerBase.__subclasses__()]
PLANNER_SUBCLASS_TYPES_AS_STR = [str(t) for t in PLANNER_SUBCLASS_TYPES]