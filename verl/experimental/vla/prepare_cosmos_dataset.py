import argparse
import os

from datasets import Dataset

DEFAULT_TASKS = [
    'move the block to the goal area',
    'align the gripper with the object',
    'push the object to the marked target',
    'place the item near the highlighted bin',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Create a tiny parquet dataset for CosmosEnv VLA/SAC demos.')
    parser.add_argument('--local_save_dir', default='~/data/cosmos_robot_rl')
    parser.add_argument('--num_train_samples', type=int, default=256)
    parser.add_argument('--num_test_samples', type=int, default=64)
    parser.add_argument('--num_tasks', type=int, default=len(DEFAULT_TASKS))
    parser.add_argument('--states_per_task', type=int, default=64)
    return parser.parse_args()


def build_split(split, num_samples, num_tasks, states_per_task):
    items = []
    task_text = DEFAULT_TASKS[:num_tasks]
    for idx in range(num_samples):
        task_id = idx % num_tasks
        state_id = idx % states_per_task
        items.append(
            {
                'data_source': split,
                'prompt': task_text[task_id],
                'state_ids': state_id,
                'task_ids': task_id,
                'ability': 'robot',
                'extra_info': {
                    'split': split,
                    'state_ids': state_id,
                    'task_ids': task_id,
                    'index': idx,
                    'task_description': task_text[task_id],
                },
            }
        )
    return Dataset.from_list(items)


def main():
    args = parse_args()
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_ds = build_split('train', args.num_train_samples, args.num_tasks, args.states_per_task)
    test_ds = build_split('test', args.num_test_samples, args.num_tasks, args.states_per_task)
    train_ds.to_parquet(os.path.join(save_dir, 'train.parquet'))
    test_ds.to_parquet(os.path.join(save_dir, 'test.parquet'))
    print(f'Wrote {len(train_ds)} train samples to {os.path.join(save_dir, "train.parquet")}')
    print(f'Wrote {len(test_ds)} test samples to {os.path.join(save_dir, "test.parquet")}')


if __name__ == '__main__':
    main()
