import argparse
import json
import os

import numpy as np
import pandas as pd
from loguru import logger

from epog.algorithm import EFS, EPoG, PlanningMetrics
from epog.algorithm.baseline.LLM_Explore import LLM_Explore
from epog.algorithm.baseline.LLM_Planner import LLM
from epog.data_gen.task_list import task_list
from epog.envs.proc_env import ProcEnv

os.environ["TOKENIZERS_PARALLELISM"] = "true"

Algorithm = EFS | EPoG | LLM | LLM_Explore


def summary_results(csv_filename: str, summary_filename: str) -> None:
    """Summarize evaluation results by task.

    Args:
        csv_filename: Path to the CSV file containing evaluation results.
        summary_filename: Path to save the summary CSV file.
    """
    df = pd.read_csv(csv_filename)

    summary_data = []
    for task_name in task_list:
        task_df = df[df["task_name"] == task_name]
        if len(task_df) == 0:
            continue

        success_rate = task_df["success"].mean() * 100
        avg_expand_node = task_df[task_df["success"] == 1]["expand_node"].mean()
        avg_travel_dist = task_df[task_df["success"] == 1]["travel_dist"].mean()

        summary_data.append(
            {
                "task_name": task_name,
                "num_scenes": len(task_df),
                "success_rate": success_rate,
                "avg_expand_node": avg_expand_node
                if not np.isnan(avg_expand_node)
                else 0,
                "avg_travel_dist": avg_travel_dist
                if not np.isnan(avg_travel_dist)
                else 0,
            }
        )

    # Add total row
    total_success_rate = df["success"].mean() * 100
    total_avg_expand = df[df["success"] == 1]["expand_node"].mean()
    total_avg_dist = df[df["success"] == 1]["travel_dist"].mean()

    summary_data.append(
        {
            "task_name": "Total",
            "num_scenes": len(df),
            "success_rate": total_success_rate,
            "avg_expand_node": total_avg_expand
            if not np.isnan(total_avg_expand)
            else 0,
            "avg_travel_dist": total_avg_dist if not np.isnan(total_avg_dist) else 0,
        }
    )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_filename, index=False)
    logger.info(f"Summary saved to {summary_filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--algorithm_name", type=str, default="LLM_Pure")
    parser.add_argument("--data_root_dir", type=str, default="data/task")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    return parser.parse_args()


def get_algorithm(env: ProcEnv, args: argparse.Namespace) -> Algorithm:
    """Get algorithm instance based on command line arguments.

    Args:
        env: The environment instance.
        args: Parsed command line arguments.

    Returns:
        Algorithm instance.

    Raises:
        ValueError: If algorithm_name is not recognized.
    """
    if args.algorithm_name == "EFS":
        return EFS(env, path_opt=True, logger=logger)
    elif args.algorithm_name == "EPoG":
        return EPoG(
            env, global_agent="llm", local_agent="llm", path_opt=True, logger=logger
        )
    elif args.algorithm_name == "LLM+Explore":
        return LLM_Explore(env, path_opt=True, logger=logger)
    elif args.algorithm_name == "LLM_Pure":
        return LLM(env, path_opt=True, logger=logger)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm_name}")


def main() -> None:
    args = parse_args()

    # Setup logging
    log_filename = os.path.join(args.work_dir, f"{args.algorithm_name}/eval.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logger.add(log_filename)

    # Setup CSV files
    csv_filename = os.path.join(args.work_dir, f"{args.algorithm_name}/eval.csv")
    summary_filename = os.path.join(args.work_dir, f"{args.algorithm_name}/summary.csv")
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(
            columns=["scene_id", "success", "expand_node", "travel_dist", "task_name"]
        )
        df.to_csv(csv_filename, index=False)

    # Run evaluation
    for task in task_list:
        data_dir = os.path.join(args.data_root_dir, task)
        scene_list_path = os.path.join(data_dir, "scene_list.json")
        with open(scene_list_path) as f:
            task_scene_ids = json.load(f)

        for scene_id in task_scene_ids:
            # Check if already evaluated
            scene_mask = df["scene_id"] == scene_id
            if scene_mask.any():
                task_names = df.loc[scene_mask, "task_name"].tolist()
                if task in task_names:
                    logger.info(f"Scene {scene_id} has been evaluated in {task}")
                    continue

            env = ProcEnv(data_dir, scene_id)
            algorithm = get_algorithm(env, args)
            try:
                algorithm.main_loop()
                result = algorithm.eval()
                logger.info(f"{args.algorithm_name} Scene {scene_id} result: {result}")
            except Exception as e:
                logger.warning(f"exception: {e}")
                result = PlanningMetrics()
            env.close()

            # Save result
            data = {
                "scene_id": scene_id,
                "success": float(result.success),
                "expand_node": result.expand_node,
                "travel_dist": result.moving_cost,
                "task_name": task,
            }
            df = pd.read_csv(csv_filename)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(csv_filename, index=False)

    summary_results(csv_filename, summary_filename)


if __name__ == "__main__":
    main()
