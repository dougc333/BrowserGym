import argparse
import time

# locally defined agent
from agent import DemoAgentArgs

# browsergym experiments utils
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--all_miniwob",
        type=str2bool,
        default=False,
        help="Run every registered MiniWoB task once.",
    )
    parser.add_argument(
        "--delay_between_tasks",
        type=float,
        default=2.0,
        help="Seconds to wait between tasks when --all_miniwob is enabled.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum agent steps per task (default: 10 for all MiniWoB, 100 otherwise).",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    return parser.parse_args()


def run_task(args, task_name):
    # setting up agent config
    agent_args = DemoAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
    )

    # setting up environment config
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=args.max_steps if args.max_steps is not None else (10 if args.all_miniwob else 100),
        headless=False,  # keep the browser open
        # viewport={"width": 1500, "height": 1280},  # can be played with if needed
    )

    # for openended task, set environment and agent to interactive chat mode on a start url
    if task_name == "openended":
        agent_args.chat_mode = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}

    # setting up the experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # running and logging results
    exp_args.prepare("./results")
    exp_args.run()

    # loading and printing results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")

    return exp_record


def main():
    print(
        """\
--- WARNING ---
This is a basic agent for demo purposes.
Visit AgentLab for more capable agents with advanced features.
https://github.com/ServiceNow/AgentLab"""
    )

    args = parse_args()

    if not args.all_miniwob:
        run_task(args, args.task_name)
        return

    from browsergym.miniwob import ALL_MINIWOB_TASKS

    task_names = [task.get_task_id() for task in ALL_MINIWOB_TASKS]
    results = []
    for index, task_name in enumerate(task_names, start=1):
        print(f"\n=== MiniWoB task {index}/{len(task_names)}: {task_name} ===")
        try:
            record = run_task(args, task_name)
            results.append((task_name, record.get("cum_reward"), None))
        except Exception as exc:
            print(f"Task failed: {task_name}: {exc}")
            results.append((task_name, None, str(exc)))

        if index < len(task_names):
            print(f"Waiting {args.delay_between_tasks:g}s before the next task...")
            time.sleep(args.delay_between_tasks)

    succeeded = sum(error is None for _, _, error in results)
    rewarded = sum(reward == 1 for _, reward, error in results if error is None)
    print(
        f"\nCompleted {len(results)} tasks: {succeeded} ran without errors, "
        f"{rewarded} received full reward."
    )


if __name__ == "__main__":
    main()
