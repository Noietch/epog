import os
import argparse
import json
import pandas as pd
from loguru import logger

from epog.envs.proc_env import ProcEnv
from epog.data_gen.task_list import task_list
from epog.algorithm import EFS, EPoG, PlanningMetrics
from epog.algorithm.baseline.LLM_Planner import LLM
from epog.algorithm.baseline.LLM_Explore import LLM_Explore

os.environ['TOKENIZERS_PARALLELISM'] = "true"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--algorithm_name', type=str, default='LLM_Pure')
    parser.add_argument('--data_root_dir', type=str, default='data/task')
    parser.add_argument('--work_dir', type=str, default='work_dir')
    args = parser.parse_args()
    return args

def get_algorithm(env, args):
    if args.algorithm_name == 'EFS':
        alogrithm = EFS(env, path_opt=True, logger=logger)
    elif args.algorithm_name == 'EPoG':
        alogrithm = EPoG(env,global_agent='llm', local_agent='llm', path_opt=True, logger=logger)
    elif args.algorithm_name == 'LLM+Explore':
        alogrithm = LLM(env, path_opt=True, logger=logger)
    elif args.algorithm_name == 'LLM_Pure':
        alogrithm = LLM_Explore(env, path_opt=True, logger=logger)
    return alogrithm

def main():
    args = parse_args()
    log_filename = os.path.join(args.work_dir, f'{args.algorithm_name}/eval.log')
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logger.add(log_filename)

    csv_filename = os.path.join(args.work_dir, f'{args.algorithm_name}/eval.csv')
    summary_filename = os.path.join(args.work_dir, f'{args.algorithm_name}/summary.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=['scene_id', 'success', 'expand_node', 'travel_dist', 'task_name'])
        df.to_csv(csv_filename, index=False)

    for task in list(task_list.keys()):
        data_dir = os.path.join(args.data_root_dir, task)
        task_scene_id = json.load(open(os.path.join(data_dir, 'scene_list.json')))
        for scene_id in task_scene_id:
            if scene_id in df['scene_id'].values and task in df[df['scene_id'] == scene_id]['task_name'].values:
                print(f"Scene {scene_id} has been evaluated in {task}")
                continue
            env = ProcEnv(data_dir, scene_id)
            alogrithm = get_algorithm(env, args)
            try:
                alogrithm.main_loop()
                result = alogrithm.eval()
                logger.info(f"{args.algorithm_name} Scene {scene_id} result: {result}")
            except Exception as e:
                print("exception", e)
                result = PlanningMetrics()
            env.close()
            
            data = {
                "scene_id": scene_id,
                "success": float(result.success),
                "expand_node": result.expand_node,
                "travel_dist": result.moving_cost,
                "task_name": task
            }
            df = pd.read_csv(csv_filename)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(csv_filename, index=False)

    summary_results(csv_filename, summary_filename)

if __name__ == '__main__':
    main()