"""
CLI training script for the offline DQN policy.

Usage:
    python -m rl.train
    python -m rl.train --log logs/recommendations.jsonl --epochs 500 --alpha 1.0
    python -m rl.train --eval-only
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train offline DQN policy")
    parser.add_argument("--log",     default="logs/recommendations.jsonl")
    parser.add_argument("--epochs",  type=int,   default=300)
    parser.add_argument("--alpha",   type=float, default=1.0, dest="cql_alpha")
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--patience",type=int,   default=40)
    parser.add_argument("--reward",  default="reward_5d_sharpe",
                        choices=["reward_5d_sharpe", "reward_calmar",
                                 "reward_10d_sharpe", "reward_regime_adj",
                                 "reward_rr_weighted"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    from rl.dqn_policy import OfflineDQNTrainer, ReplayBuffer

    buf = ReplayBuffer.from_log_file(args.log, reward_key=args.reward)
    print(f"Loaded {len(buf)} labelled records from {args.log} (reward={args.reward})")

    if len(buf) < 10:
        print("ERROR: need ≥10 labelled records. Run fetch_and_label_outcomes() first.")
        return

    trainer = OfflineDQNTrainer(
        lr=args.lr,
        cql_alpha=args.cql_alpha,
        batch_size=args.batch,
        patience=args.patience,
    )

    if args.eval_only:
        result = trainer.evaluate(buf)
        print("Evaluation results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    print(f"Training for up to {args.epochs} epochs …")
    result = trainer.train(buf, n_epochs=args.epochs)
    print(f"\nTraining complete.")
    print(f"  Best val loss : {result['best_val_loss']:.6f}")
    print(f"  Epochs run    : {result['epochs_run']}")
    print(f"  Checkpoint    : rl/checkpoints/best.pt")

    # Evaluate on full buffer
    eval_result = trainer.evaluate(buf)
    print("\nPost-training evaluation:")
    for k, v in eval_result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
