import optuna
import argparse

# Define the objective function
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    return (x - 2)**2 + (y + 3)**2

def create_sampler(sampler_name, search_space=None, seed=None):
    if sampler_name == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "GPSampler":
        return optuna.samplers.GPSampler(seed=seed)
    elif sampler_name == "PartialFixedSampler":
        # Example: {'x': 0} fixed_params and optuna.samplers.RandomSampler() as base_sampler
        fixed_params = {'x': 0}
        base_sampler = optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.PartialFixedSampler(fixed_params, base_sampler)
    elif sampler_name == "NSGAIISampler":
        return optuna.samplers.NSGAIISampler(seed=seed)
    elif sampler_name == "QMCSampler":
        return optuna.samplers.QMCSampler(seed=seed)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

def main():
    parser = argparse.ArgumentParser(description="Optuna CLI for hyperparameter optimization.")
    parser.add_argument("--sampler", type=str, required=True, help="Sampler to use for optimization.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials for optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna storage.")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the study.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout for the optimization in seconds.")
    args = parser.parse_args()

    sampler = create_sampler(args.sampler, args.seed)

    if args.storage:
        storage = optuna.storages.RDBStorage(args.storage)
    else:
        storage = None

    study = optuna.create_study(study_name=args.study_name, direction="minimize", sampler=sampler, storage=storage)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best value (loss): ", study.best_value)
    print("Best parameters: ", study.best_params)

if __name__ == "__main__":
    main()