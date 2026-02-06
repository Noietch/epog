# Some functions are modified from searchProblem.py - representations of search problems http://aipython.org
import heapq
import logging

from pog.planning.problem import PlanningOnGraphPath, PlanningOnGraphProblem


class Searcher:
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    This does depth-first search unless overridden
    """

    def __init__(self, problem, find_min_path=False, pruned=True):
        """creates a searcher from a problem"""
        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0
        self.add_to_frontier(PlanningOnGraphPath(problem.start_node()))
        self.solutions = []
        self.min_cost_path = None
        self.find_min_path = find_min_path
        self.pruned = pruned

    def initialize_frontier(self):
        self.frontier = []

    def empty_frontier(self):
        return self.frontier == []

    def add_to_frontier(self, path):
        self.frontier.append(path)

    def search(self):
        """returns (next) path from the problem's start node
        to a goal node.
        Returns None if no path exists.
        """
        min_planning_cost = float("inf")
        expanded = 0
        while not self.empty_frontier():
            expanded += 1
            if expanded % 1000 == 0:
                logging.info(f"Expanded: {expanded}")
            if expanded > 10000:
                logging.warning("Expanded more than 10000")
                break
            while not self.empty_frontier():
                path: PlanningOnGraphPath = self.frontier.pop()
                if path.cost < min_planning_cost or not self.pruned:
                    break
            self.num_expanded += 1
            if self.problem.is_goal(path.end()):  # solution found
                self.solutions.append(path)  # store the solution found
                if not self.find_min_path:
                    self.min_cost_path = path
                    return expanded
                if path.cost < min_planning_cost:
                    min_planning_cost = path.cost
                    self.min_cost_path = path
            else:
                neighs = self.problem.neighbors(path.end())
                for arc in reversed(list(neighs)):
                    self.add_to_frontier(PlanningOnGraphPath(path, arc))
        return expanded


class FrontierPQ:
    """A frontier consists of a priority queue (heap), frontierpq, of
        (value, index, path) triples, where
    * value is the value we want to minimize (e.g., path cost + h).
    * index is a unique index for each element
    * path is the path on the queue
    Note that the priority queue always returns the smallest element.
    """

    def __init__(self):
        """constructs the frontier, initially an empty priority queue"""
        self.frontier_index = 0  # the number of items ever added to the frontier
        self.frontierpq = []  # the frontier priority queue

    def empty(self):
        """is True if the priority queue is empty"""
        return self.frontierpq == []

    def add(self, path, value):
        """add a path to the priority queue
        value is the value to be minimized"""
        self.frontier_index += 1  # get a new unique index
        heapq.heappush(self.frontierpq, (value, -self.frontier_index, path))

    def pop(self):
        """returns and removes the path of the frontier with minimum value."""
        (_, _, path) = heapq.heappop(self.frontierpq)
        return path

    def count(self, val):
        """returns the number of elements of the frontier with value=val"""
        return sum(1 for e in self.frontierpq if e[0] == val)

    def __repr__(self):
        """string representation of the frontier"""
        return str([(n, c, str(p)) for (n, c, p) in self.frontierpq])

    def __len__(self):
        """length of the frontier"""
        return len(self.frontierpq)

    def __iter__(self):
        """iterate through the paths in the frontier"""
        for _, _, path in self.frontierpq:
            yield path


class AStarSearcher(Searcher):
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    """

    def __init__(self, problem):
        super().__init__(problem)

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def empty_frontier(self):
        return self.frontier.empty()

    def add_to_frontier(self, path):
        """add path to the frontier with the appropriate cost"""
        value = path.cost + self.problem.heuristic(path.end())
        self.frontier.add(path, value)


def test(
    SearchClass: Searcher,
    problem: PlanningOnGraphProblem,
    find_min_path=False,
    pruned: bool = False,
):
    """Unit test for aipython searching algorithms.
    SearchClass is a class that takes a problemm and implements search()
    problem is a search problem
    solutions is a list of optimal solutions
    """
    schr: Searcher = SearchClass(problem, pruned=pruned, find_min_path=find_min_path)
    expanded = schr.search()
    return schr.min_cost_path, expanded
