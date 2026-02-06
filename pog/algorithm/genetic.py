import copy
import logging
import multiprocessing as mp
import random
import time

import networkx as nx
import vedo
from numpy.random import rand

from pog.algorithm.params import (
    AREA_RATIO_MARGIN,
    CROSSOVER_CHILDREN,
    FITNESS_THRESH,
    MAX_GENERATION,
    MAX_POPULATION,
    MUTATION_PROBABILITY,
    TOURNAMENT_K,
)
from pog.algorithm.structure import simulated_annealing_structure
from pog.algorithm.utils import to_node_lite
from pog.graph import node, shape
from pog.graph.chromosome import Chromosome, Gene, to_graph


def computeAreaHeuristic(chromo):
    """Compute area heuristic

    Args:
        chromo (Chromosome): Chromosome to compute heuristic

    Returns:
        satisfy (bool): True if all area constraints are satisfied
        cost (int): Accumulated violation cost
    """
    chromo.trackDepth()
    max_depth = len(nx.algorithms.dag_longest_path(chromo.chromograph))
    cost = 0
    satisfy = True
    for depth in range(max_depth):
        for node in chromo.depth_dict[depth]:
            ratio = {}
            n_succ = {}
            for succ in chromo.chromograph.successors(node):
                if chromo.chromosome[succ].parent_affordance_name not in n_succ:
                    n_succ[chromo.chromosome[succ].parent_affordance_name] = 1
                else:
                    n_succ[chromo.chromosome[succ].parent_affordance_name] += 1

                area_2 = chromo.node_dict[succ].affordance[
                    chromo.chromosome[succ].child_affordance_name
                ]["area"]
                area_1 = chromo.node_dict[node].affordance[
                    chromo.chromosome[succ].parent_affordance_name
                ]["area"]

                if chromo.chromosome[succ].parent_affordance_name not in ratio:
                    ratio[chromo.chromosome[succ].parent_affordance_name] = (
                        area_2 / area_1
                    )
                else:
                    ratio[chromo.chromosome[succ].parent_affordance_name] += (
                        area_2 / area_1
                    )

            for key, value in ratio.items():
                if value > AREA_RATIO_MARGIN and n_succ[key] > 1:
                    satisfy = False

                cost += max(value, AREA_RATIO_MARGIN) - AREA_RATIO_MARGIN

    return satisfy, cost


def computeFitness(chromo: Chromosome, method="heuristic", **kwargs):
    """Compute fitness of chromosome, the lower, the better.

    Args:
        chromo (Chromosome): Chromosome to compute fitness
        method (str): Method of computing heuristics

    Returns:
        chromo.cnt_sat (bool): True if all constraints are satisfied
        chromo.fitness (int): Fitness of current chromosome
    """
    if method == "structure":  # Extremly time-consuming
        _, stable_cnt_sat, stable_cost = simulated_annealing_structure(
            chromo.to_graph(), fixed_nodes=chromo.fixed_nodes, **kwargs
        )
        prop_cnt_sat, prop_cost = chromo.checkPropConstraints()
        height_cnt_sat, height_cost = chromo.checkHeightConstraints()
        chromo.fitness = stable_cost + prop_cost + height_cost
        chromo.cnt_sat = stable_cnt_sat and prop_cnt_sat and height_cnt_sat
    elif method == "heuristic":
        cnt_sat, cost = computeAreaHeuristic(chromo)
        prop_cnt_sat, prop_cost = chromo.checkPropConstraints()
        height_cnt_sat, height_cost = chromo.checkHeightConstraints()
        cont_cnt_sat, cont_cost = chromo.checkContConstraints()
        chromo.fitness = cost + prop_cost + height_cost + cont_cost
        chromo.cnt_sat = cnt_sat and prop_cnt_sat and height_cnt_sat and cont_cnt_sat
    else:
        logging.error(
            f"Unsupported method {method}. Supported method: structure or heuristic."
        )
    return chromo.cnt_sat, chromo.fitness


def tournament_selection(population):
    selected_population = []
    for _i in range(MAX_POPULATION):
        candidates = random.sample(population, TOURNAMENT_K)
        candidates.sort()
        selected_population.append(candidates[0])
    return selected_population


def genetic_programming(**kwargs):
    """genetic programming for structure search

        NOTE: Multiprocessing only work for 'heuritic' method.

    Returns:
        (chromosome): Best fit chromosome
    """
    population = []
    method = kwargs.get("method", "heuristic")
    multiprocess = kwargs.get("multiprocess", False)

    for _idx in range(MAX_POPULATION):
        population.append(Chromosome(**kwargs))
        if population[-1].initialize():
            computeFitness(population[-1], method)
        else:
            population.pop()

    population.sort()

    if multiprocess:
        logging.info("Start multi-processing.")
    else:
        logging.info("Start single-processing.")

    for n_gen in range(MAX_GENERATION):
        start = time.time()
        logging.debug(f"Generation: {n_gen + 1}")
        population = tournament_selection(population)
        temp_population = []
        if multiprocess:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                temp_population = pool.map(
                    genetic_programming_helper,
                    [
                        population[i * mp.cpu_count() : (i + 1) * mp.cpu_count()]
                        for i in range(
                            (len(population) + mp.cpu_count() - 1) // mp.cpu_count()
                        )
                    ],
                )
            population = temp_population[0]
        else:
            for chromo in population:
                for _i in range(CROSSOVER_CHILDREN):
                    new_chromo = copy.deepcopy(chromo)
                    new_chromo.crossover()
                    if rand() < MUTATION_PROBABILITY:
                        new_chromo.mutate()
                    _, _ = computeFitness(new_chromo, method)
                    temp_population.append(new_chromo)
            population = temp_population
        population.sort()
        population_sat = [ch for ch in population if ch.cnt_sat]
        end = time.time()
        if population_sat and population_sat[0].fitness < FITNESS_THRESH:
            logging.info(
                f"Finished generation {n_gen + 1} in {end - start:.4f} seconds. Solution Found! Fitness: {population_sat[0].fitness:.4f}."
            )
            return population_sat[0]
        else:
            logging.info(
                f"Finished generation {n_gen + 1} in {end - start:.4f} seconds. Best fitness: {population[0].fitness:.4f}."
            )
    return population[0]


def genetic_programming_helper(population, method="heuristic"):
    temp_population = []
    for chromo in population:
        for _i in range(CROSSOVER_CHILDREN):
            new_chromo = copy.deepcopy(chromo)
            new_chromo.crossover()
            if rand() > MUTATION_PROBABILITY:
                new_chromo.mutate()
            computeFitness(new_chromo, method)
            temp_population.append(new_chromo)
    return temp_population


if __name__ == "__main__":
    node_dict = {}
    node0 = shape.Box(size=[0.5, 0.5, 0.05])
    node_node0 = node.Node(id=0, shape=node0)

    node1 = shape.Sphere(radius=0.1)
    node_node1 = node.Node(id=1, shape=node1)

    node2 = shape.Box(size=(0.1, 0.2, 0.3))
    node_node2 = node.Node(id=2, shape=node2)

    node3 = shape.Cylinder(height=0.3, radius=0.1)
    node_node3 = node.Node(id=3, shape=node3)

    node4 = shape.Cylinder(height=0.1, radius=0.05)
    node_node4 = node.Node(id=4, shape=node4)

    node5 = shape.Box(size=0.06)
    node_node5 = node.Node(id=5, shape=node5)

    node6 = shape.Box(size=0.09)
    node_node6 = node.Node(id=6, shape=node6)

    node7 = shape.Cylinder(height=0.15, radius=0.05)
    node_node7 = node.Node(id=7, shape=node7)

    node8 = shape.Imported(file_path="./pog_example/mesh/pan.stl")
    node_node8 = node.Node(id=8, shape=node8)

    node_dict[0] = node_node0
    node_dict[1] = node_node1
    node_dict[2] = node_node2
    node_dict[3] = node_node3
    node_dict[4] = node_node4
    node_dict[5] = node_node5
    node_dict[6] = node_node6
    node_dict[7] = node_node7
    node_dict[8] = node_node8

    node_dict_lite = to_node_lite(node_dict)

    constraints = {}
    cnt = {
        2: Gene(
            node_dict_lite[0].affordance["box_aff_pz"],
            node_dict_lite[2].affordance["box_aff_nx"],
        )
    }
    constraints["gene"] = cnt
    # constraints['fixed']= [0,3]
    constraints["height"] = 0.4
    # constraints['propagation'] = {4:2}

    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("test.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    best = genetic_programming(
        node_dict_lite=node_dict_lite, constraints=constraints, method="heuristic"
    )

    print(best)
    g = to_graph(best, node_dict)
    start = time.time()
    g, cnt_sat, cost = simulated_annealing_structure(g)
    end = time.time()
    print(
        f"Finished optimization in {end - start:.4f} seconds. Cost: {cost}; CNT_SAT: {cnt_sat} "
    )
    g.create_scene()
    vedo.show(g.scene.dump(concatenate=True), axes=1)
