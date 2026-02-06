import logging
import time
from copy import deepcopy

import numpy as np
import vedo
from numpy.random import rand, randn

from pog.algorithm.params import (
    EPS_THRESH,
    MAX_ITER,
    MAX_STEP_SIZE,
    MAX_TEMP,
    NUM_CYCLES_EPS,
    NUM_CYCLES_STEP,
    NUM_CYCLES_TEMP,
    SAVE_HISTORY,
    STEP_SIZE,
    TEMP_COEFF,
)
from pog.algorithm.utils import (
    arr2pose,
    checkConstraints,
    gen_bound,
    objective,
    pose2arr,
)
from pog.graph.graph import Graph, createTestGraph


# simulated annealing algorithm
def simulated_annealing(
    objective,
    sg: Graph,
    node_id=None,
    random_start=False,
    verbose=False,
    method="standard",
):
    """simulated annealing algorithm to maximum stability.

            NOTE: This function is only for scene graph with max depth = 2

    Args:
            objective (list): a list of objective functions
            sg (Graph): scene graph
            node_id (list, optional): a list of nodes to be optimized. Defaults to None.
            random_start (bool, optional): Randomly select initial configuration. Defaults to False.
            verbose (bool, optional): More outputs. Defaults to False.

    Returns:
            best: best configuration
            best_eval: cost of best configuration
    """
    if method == "adaptive":
        return adaptive_simulated_annealing(
            objective, sg, node_id, random_start=random_start, verbose=verbose
        )

    pose = sg.getPose(edge_id=node_id) if node_id is not None else sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval, _ = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    # run the algorithm
    t = MAX_TEMP
    best_eval_arr = []
    history = []
    for i in range(MAX_ITER):
        # take a step
        step_direction = randn(len(curr))  # Gaussian
        # step_direction = 2. * (rand(len(curr)) - 0.5) # Uniform
        candidate = curr + step_direction / np.linalg.norm(step_direction) * STEP_SIZE

        # evaluate candidate point
        pose = arr2pose(candidate, object_pose_dict, pose)
        sg.setPose(pose)
        candidate_eval, _ = objective[0](candidate, object_pose_dict, sg)
        if SAVE_HISTORY:
            history.append(deepcopy(pose))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate temperature for current epoch
        # t = temp / float(i + 1)
        t = t * TEMP_COEFF

        # calculate metropolis acceptance criterion
        metropolis = np.exp(min(-diff / t, 700.0))  # <- avoid overflow

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            if verbose:
                best_tmp = [float(f"{x:.4f}") for x in list(best)]
                logging.debug(f">{i} f({best_tmp}) = {best_eval:.4f}, temp: {t:.4f}")

            best_eval_arr.append(diff)
            if (
                len(best_eval_arr) > NUM_CYCLES_EPS
                and (abs(np.array(best_eval_arr[-NUM_CYCLES_EPS:])) < EPS_THRESH).all()
            ):
                break

        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # if diff < 0:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


def adaptive_simulated_annealing(
    objective, sg: Graph, node_id=None, random_start=False, verbose=False
):
    pose = sg.getPose(edge_id=node_id) if node_id is not None else sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    eval_arr, counts_cycles, counts_resets = [], 0, 0
    n = len(curr)
    a = np.zeros(n)
    step_vector = MAX_STEP_SIZE * np.ones(n)
    c = 0.1 * np.ones(n)
    t = MAX_TEMP
    # run the algorithm
    i = 0
    while i < 5000:
        i += 1
        for iter in range(n):
            step = np.zeros(n)
            step[iter] = 2 * (rand() - 0.5) * step_vector[iter]
            temp = curr + step
            temp_eval = objective[0](temp, object_pose_dict, sg)
            diff_temp_eval = temp_eval - curr_eval
            if diff_temp_eval < 0 or rand() < np.exp(-diff_temp_eval / t):
                curr, curr_eval = temp, temp_eval
                a[iter] += 1.0
                if curr_eval < best_eval:
                    best, best_eval = curr, curr_eval

        counts_cycles += 1
        if counts_cycles <= NUM_CYCLES_STEP:
            continue

        counts_cycles = 0
        step_vector = corana_update(step_vector, a, c, NUM_CYCLES_STEP)
        a = np.zeros(n)
        counts_resets += 1
        if counts_resets <= NUM_CYCLES_TEMP:
            continue

        t *= TEMP_COEFF
        counts_resets = 0
        eval_arr.append(curr_eval)
        if not (
            len(eval_arr) > NUM_CYCLES_EPS
            and eval_arr[-1] - best_eval <= EPS_THRESH
            and (
                abs((eval_arr[-1] - np.array(eval_arr))[-NUM_CYCLES_EPS:]) <= EPS_THRESH
            ).all()
        ):
            curr, curr_eval = best, best_eval
            if verbose:
                best_tmp = [float(f"{x:.4f}") for x in list(best)]
                logging.debug(f">{i} f({best_tmp}) = {best_eval:.4f}, temp: {t:.4f}")

        else:
            break

    return [best, best_eval]


def corana_update(v, a, c, ns):
    for i in range(len(v)):
        ai, ci = a[i], c[i]

        if ai > 0.6 * ns:
            v[i] *= 1 + ci * (ai / ns - 0.6) / 0.4
        elif ai < 0.4 * ns:
            v[i] /= 1 + ci * (0.4 - ai / ns) / 0.4

    return v


if __name__ == "__main__":
    g = Graph("test scene", fn=createTestGraph)
    sg = g.getSubGraph(2)
    pose = sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)
    start = time.time()
    best, best_eval = simulated_annealing(
        [objective], sg, random_start=True, verbose=True
    )
    end = time.time()
    print(f"Run {MAX_ITER} iterations in {end - start:.4f} seconds")
    sg.setPose(arr2pose(best, object_pose_dict, pose))
    print(checkConstraints([], object_pose_dict, sg))
    sg.create_scene()
    vedo.show(sg.scene.dump(concatenate=True), axes=1)
