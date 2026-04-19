import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from flask import Flask, request

# Internal Project Imports for Prediction Logic
from src.core.project_config import ProjectModelType, ProjectPaths
from src.data_collection.box_office_collector import BoxOfficeCollector
from src.data_collection.review_collector import ReviewCollector, TargetWebsite
from src.data_handling.box_office import BoxOffice
from src.data_handling.file_io import YamlFile
from src.data_handling.movie_collections import MovieData
from src.data_handling.reviews import PublicReview
from src.models.box_office_classification.components.data_processor import BoxOfficeClassificationDataProcessor
from src.models.box_office_classification.components.model_core import (
    BoxOfficeClassificationModelCore,
    BoxOfficeClassificationPredictParams,
)
from src.models.box_office_common.box_office_data_processor import BoxOfficeDataConfig


@dataclass
class ComputeRequest:
    """
    Represents a computation request from the client.

    :ivar command: The specific command to execute.
    :ivar arguments: A dictionary of arguments required for the command.
    """

    command: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'ComputeRequest':
        """
        Deserialize a JSON string into a ComputeRequest object.

        :param json_str: The JSON string to parse.
        :return: An instance of ComputeRequest.
        """
        try:
            data: dict[str, Any] = json.loads(json_str)
            return cls(
                command=data.get("command", ""),
                arguments=data.get("arguments", {})
            )
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")


@dataclass
class ComputeResponse:
    """
    Represents the response sent back to the client.

    :ivar status: The status of the request ('in_progress', 'completed', 'failed').
    :ivar result: The final output of the computation.
    :ivar progress: Information about the current progress.
    :ivar error: Error details if the request failed.
    """

    status: str
    result: Optional[dict[str, Any]] = None
    progress: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None

    def to_json(self) -> str:
        """
        Serialize the ComputeResponse object to a JSON string.

        :return: A JSON formatted string.
        """
        return json.dumps(asdict(self), ensure_ascii=False)


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

app: Flask = Flask(__name__)


class MovieBoxOfficePredictor:
    """
    Encapsulates the prediction logic for movie box office classification.
    """

    def __init__(self, model_id: str = "default_model", epoch: int = 10) -> None:
        """
        Initializes the predictor with model settings.

        :param model_id: The ID of the model to use.
        :param epoch: The epoch number of the model file.
        """
        self._model_id: str = model_id
        self._epoch: int = epoch
        self._model_type: ProjectModelType = ProjectModelType.BOX_OFFICE_CLASSIFICATION

    def predict_rating_with_data(self, movie_data: MovieData) -> int:
        """
        Perform prediction using already fetched MovieData.

        :param movie_data: The movie data instance.
        :return: Predicted class index.
        """
        try:
            artifacts_folder: Path = ProjectPaths.get_model_root_path(
                model_id=self._model_id, model_type=self._model_type
            )
            config_path: Path = artifacts_folder / "config.yaml"
            if not config_path.exists():
                logger.warning(f"Config not found for {self._model_id}, returning fallback.")
                return 1

            original_config: dict[str, Any] = YamlFile(path=config_path).load_single_document()
            model_file_path: Path = artifacts_folder / f"{self._model_id}_{self._epoch:04d}.keras"

            data_processor: BoxOfficeClassificationDataProcessor = BoxOfficeClassificationDataProcessor(
                model_artifacts_path=artifacts_folder
            )
            model_core: BoxOfficeClassificationModelCore = BoxOfficeClassificationModelCore(model_path=model_file_path)

            processing_config: BoxOfficeDataConfig = BoxOfficeDataConfig(
                training_week_len=original_config.get('training_week_len', 4)
            )
            processed_input: np.ndarray = data_processor.process_for_prediction(
                single_input=movie_data, config=processing_config
            )

            pred_params: BoxOfficeClassificationPredictParams = BoxOfficeClassificationPredictParams(verbose=0)
            probabilities: np.ndarray = model_core.predict(data=processed_input, params=pred_params)

            return int(np.argmax(probabilities, axis=1)[0])
        except Exception as e:
            logger.error(f"Prediction logic error for {movie_data.name}: {e}")
            return 0


def _fetch_movie_data_online(movie_name: str) -> MovieData:
    """
    Fetches movie data (box office and reviews) from online sources.

    :param movie_name: The name of the movie.
    :return: A MovieData instance.
    """
    logger.info(f"Fetching online data for movie: {movie_name}")
    box_office_history: list[BoxOffice] = []
    try:
        with BoxOfficeCollector(download_mode='WEEK', headless=True) as collector:
            box_office_history, _ = collector.fetch_single_movie_data(movie_name=movie_name)
    except Exception as e:
        logger.warning(f"Failed to fetch box office data online for '{movie_name}': {e}")

    all_public_reviews: list[PublicReview] = []
    try:
        ptt_collector: ReviewCollector = ReviewCollector(target_website=TargetWebsite.PTT)
        all_public_reviews.extend(ptt_collector.collect_reviews_for_movie(movie_name=movie_name))
    except Exception as e:
        logger.warning(f"Failed to fetch PTT reviews online for '{movie_name}': {e}")

    return MovieData(
        id=-1,
        name=movie_name,
        box_office=box_office_history,
        public_reviews=all_public_reviews,
        expert_reviews=[]
    )


def verify_tensorflow_environment() -> dict[str, Any]:
    """
    Verify the TensorFlow installation and GPU availability.

    :return: A dictionary containing version, GPU status, and a calculation test result.
    """
    try:
        logger.info("Starting TensorFlow verification...")

        version: str = tf.__version__
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_available: bool = len(gpu_devices) > 0
        gpu_details: list[str] = [str(d) for d in gpu_devices]

        # Simple matrix multiplication test
        a: tf.Tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b: tf.Tensor = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c: tf.Tensor = tf.matmul(a, b)

        return {
            "version": version,
            "gpu_available": gpu_available,
            "gpu_details": gpu_details,
            "tensor_calculation_test": f"Matmul result shape: {c.shape}",
            "operation_success": True
        }
    except Exception as e:
        logger.error(f"TensorFlow verification failed: {e}")
        return {
            "operation_success": False,
            "error": str(e)
        }


def get_movie_data_logic(movie_name: str) -> dict[str, Any]:
    """
    Retrieve movie information, perform AI prediction, and fetch real reviews.
    """
    # 1. Fetch real-time data from online sources
    fetched_data: MovieData = _fetch_movie_data_online(movie_name=movie_name)

    # 2. Perform AI prediction using the fetched data
    predictor: MovieBoxOfficePredictor = MovieBoxOfficePredictor()
    predict_result: int = predictor.predict_rating_with_data(movie_data=fetched_data)

    # 3. Extract the first 10 real reviews from the fetched data
    real_reviews: list[str] = [r.content for r in fetched_data.public_reviews[:10]]

    # 4. Mock additional details (In a real app, these would come from a DB)
    mock_db_row: dict[str, Any] = {
        "id": fetched_data.id,
        "chinese_name": movie_name,
        "english_name": "Dynamic Title from Web",
        "num_of_reviews": len(fetched_data.public_reviews),
        "box_office_amount": fetched_data.box_office[-1].amount if fetched_data.box_office else 0,
        "box_office_amount_last_week": fetched_data.box_office[-2].amount if len(fetched_data.box_office) > 1 else 0,
        "photo_path": "/static/posters/default.jpg",
        "introduction": "This introduction could also be fetched from the review summaries...",
        "director": "Director Name",
        "scenarist": "Scenarist Name",
    }

    return {
        "id": mock_db_row["id"],
        "chinese_name": mock_db_row["chinese_name"],
        "english_name": mock_db_row["english_name"],
        "num_of_reviews": mock_db_row["num_of_reviews"],
        "box_office_amount": mock_db_row["box_office_amount"],
        "box_office_amount_last_week": mock_db_row["box_office_amount_last_week"],
        "photo_path": mock_db_row["photo_path"],
        "introduction": mock_db_row["introduction"],
        "director": mock_db_row["director"],
        "scenarist": mock_db_row["scenarist"],
        "actors": ["Actor A", "Actor B"],
        "reviews": real_reviews,  # Using the 10 real reviews here
        "predict_result": predict_result
    }


@app.route('/compute', methods=['POST'])
def compute():
    """
    Handle POST requests to the /compute endpoint.

    :return: A Flask response object with the operation results.
    """
    try:
        raw_data: str = request.get_data(as_text=True)
        if not raw_data:
            resp: ComputeResponse = ComputeResponse(status="failed", error={"message": "No JSON data received"})
            return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

        try:
            req_obj: ComputeRequest = ComputeRequest.from_json(json_str=raw_data)
        except ValueError as e:
            resp: ComputeResponse = ComputeResponse(status="failed", error={"message": str(e)})
            return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

        if req_obj.command == "verify_tensorflow":
            result: dict[str, Any] = verify_tensorflow_environment()
            return app.response_class(response=ComputeResponse(status="completed", result=result).to_json(), status=200, mimetype='application/json')

        elif req_obj.command == "get_movie_data":
            movie_name: str = req_obj.arguments.get("movie_name", "")
            if not movie_name:
                resp = ComputeResponse(status="failed", error={"message": "Argument 'movie_name' is required"})
                return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

            result = get_movie_data_logic(movie_name=movie_name)
            return app.response_class(response=ComputeResponse(status="completed", result=result).to_json(), status=200, mimetype='application/json')

        resp = ComputeResponse(status="failed", error={"message": f"Unknown command: {req_obj.command}"})
        return app.response_class(response=resp.to_json(), status=400, mimetype='application/json')

    except Exception as e:
        logger.error(f"Server error: {e}")
        resp = ComputeResponse(status="failed", error={"message": str(e)})
        return app.response_class(response=resp.to_json(), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11100, debug=True)
