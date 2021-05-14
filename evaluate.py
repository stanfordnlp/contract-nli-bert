import json
import logging

import click

from contract_nli.evaluation import evaluate_all

logger = logging.getLogger(__name__)


@click.command()
@click.argument('task', type=str)
@click.argument('dataset-path', type=click.Path(exists=True))
@click.argument('prediction-path', type=click.Path(exists=True))
@click.argument('output-path', type=str)
def main(task, dataset_path, prediction_path, output_path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    with open(dataset_path) as fin:
        dataset = json.load(fin)
    with open(prediction_path) as fin:
        prediction = json.load(fin)
    metrics = evaluate_all(dataset, prediction,
                           [1, 3, 5, 8, 10, 15, 20, 30, 40, 50],
                           task)
    logger.info(f"Results@: {json.dumps(metrics, indent=2)}")
    with open(output_path, 'w') as fout:
        json.dump(metrics, fout, indent=2)


if __name__ == "__main__":
    main()
