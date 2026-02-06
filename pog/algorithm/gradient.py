import logging

import numpy as np
from numpy.random import rand, randn

from pog.algorithm.params import (
    EPS_THRESH,
    MAX_ITER,
    NUM_CYCLES_EPS,
    STEP_SIZE,
    TEMP_COEFF,
)
from pog.algorithm.utils import arr2pose, gen_bound, pose2arr
from pog.graph.graph import Graph


# simulated annealing algorithm
def gradient_descent(
    objective,
    sg: Graph,
    node_id=None,
    random_start=False,
    verbose=False,
):
    pose = sg.getPose(edge_id=node_id) if node_id is not None else sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        sg.setPose(arr2pose(best, object_pose_dict, pose))
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval, step_direction = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    # run the algorithm
    i = 0
    best_eval_arr = []
    t = 1
    while i < MAX_ITER:
        i += 1
        # take a step
        gauss_noise = t * randn(len(curr))  # Gaussian
        t = t * TEMP_COEFF
        candidate = curr + (step_direction + gauss_noise) * STEP_SIZE

        # evaluate candidate point
        pose = arr2pose(candidate, object_pose_dict, pose)
        sg.setPose(pose)
        candidate_eval, temp_step_direction = objective[0](
            candidate, object_pose_dict, sg
        )

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate metropolis acceptance criterion

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            if verbose:
                best_tmp = [float(f"{x:.4f}") for x in list(best)]
                logging.debug(f">{i} f({best_tmp}) = {best_eval:.4f}")

            best_eval_arr.append(diff)
            if (
                len(best_eval_arr) > NUM_CYCLES_EPS
                and (abs(np.array(best_eval_arr[-NUM_CYCLES_EPS:])) < EPS_THRESH).all()
            ):
                break

        if diff < 0:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
            step_direction = temp_step_direction

    return best, best_eval
