"""Train all models. Run after data is loaded and embedded."""
from app.core.logging import configure_logging, get_logger
from app.ml.models import anomaly, churn, recommend

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    log.info("=== training churn ===")
    churn.train()
    log.info("=== training anomaly ===")
    anomaly.train()
    log.info("=== training recommender ===")
    recommend.train()
    log.info("all models trained")


if __name__ == "__main__":
    main()
