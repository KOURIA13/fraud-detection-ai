from __future__ import annotations

from src.data_generation import FraudConfig, generate_checkout_data
from src.features import build_features
from src.rules import apply_business_rules
from src.model import run_isolation_forest
from src.scoring import combine_scores, evaluate, save_output


def main() -> None:
    cfg = FraudConfig()

    df = generate_checkout_data(cfg)
    df = build_features(df)
    df = apply_business_rules(df)
    df = run_isolation_forest(
        df=df,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    df = combine_scores(df)

    evaluate(df)
    save_output(df, output_dir=cfg.output_dir, output_file=cfg.output_file)


if __name__ == "__main__":
    main()