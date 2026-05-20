from src.monitor_model_performance import (
    monitor_performance
)

from src.check_retraining_needed import (
    check_retraining_needed
)

from src.retrain import (
    ModelRetraining
)

from src.model_comparison import (
    ModelComparison
)

from src.preprocessing import (
    DataPreprocessing
)


def run_pipeline():

    monitor_performance()

    if not check_retraining_needed():

        print(
            "\nPipeline stopped"
        )

        return
    
    print(
    "\nGenerating processed datasets"
    )

    processor = (
        DataPreprocessing()
    )

    processor.run()

    print(
        "\nStarting retraining"
    )

    retrainer = (
        ModelRetraining()
    )

    retrainer.run()

    print(
        "\nComparing models"
    )

    comparison = (
        ModelComparison()
    )

    comparison.run()

    print(
        "\nPipeline completed"
    )


if __name__ == "__main__":

    run_pipeline()