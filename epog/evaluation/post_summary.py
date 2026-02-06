import numpy as np
import pandas as pd

from epog.data_gen.task_list import task_list

efs = pd.read_csv("work_dir/EFS/eval.csv")
epog = pd.read_csv("work_dir/EPoG/eval.csv")
llm_explore = pd.read_csv("work_dir/LLM+Explore/eval.csv")
llm_pure = pd.read_csv("work_dir/LLM_Pure/eval.csv")


def diff(base: pd.DataFrame, df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    data = {
        "task_name": [],
        "expand_node_diff_percentage": [],
        "travel_dist_diff_percentage": [],
    }
    for task in list(task_list.keys()):
        task_df = df[df["task_name"] == task]
        task_base = base[base["task_name"] == task]
        expand_node_diff_percentage = []
        travel_dist_diff_percentage = []
        for item_df in task_df.itertuples():
            if item_df.success:
                base_item = task_base[task_base["scene_id"] == item_df.scene_id]
                expand_node_diff_percentage.append(
                    (item_df.expand_node - base_item.expand_node)
                    / base_item.expand_node
                    * 100
                )
                travel_dist_diff_percentage.append(
                    (item_df.travel_dist - base_item.travel_dist)
                    / base_item.travel_dist
                    * 100
                )
        data["task_name"].append(task)
        data["expand_node_diff_percentage"].append(np.mean(expand_node_diff_percentage))
        data["travel_dist_diff_percentage"].append(np.mean(travel_dist_diff_percentage))
    data["task_name"].append("average")
    data["expand_node_diff_percentage"].append(
        np.mean(data["expand_node_diff_percentage"])
    )
    data["travel_dist_diff_percentage"].append(
        np.mean(data["travel_dist_diff_percentage"])
    )
    diff_df = pd.DataFrame(data)
    diff_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # diff(efs, epog, 'work_dir/EPoG/summary_repect_to_efs.csv')
    # diff(efs, llm_explore, 'work_dir/LLM+Explore/summary_repect_to_efs.csv')
    diff(efs, llm_pure, "work_dir/LLM_Pure/summary_repect_to_efs.csv")
