from flask import (
    Flask,
    render_template,
    request
)

from src.download_artifacts import (
    download_artifacts
)

# Download latest artifacts
download_artifacts()

from src.predict import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None

    if request.method == "POST":

        try:

            input_data = {

                "hours_studied":
                float(
                    request.form.get(
                        "hours_studied"
                    )
                ),

                "previous_scores":
                float(
                    request.form.get(
                        "previous_scores"
                    )
                ),

                "extracurricular_activities":
                request.form.get(
                    "extracurricular_activities"
                ),

                "sleep_hours":
                float(
                    request.form.get(
                        "sleep_hours"
                    )
                ),

                "sample_question_papers_practiced":
                float(
                    request.form.get(
                        "sample_question_papers_practiced"
                    )
                )
            }

            predictor = PredictionPipeline()

            prediction = predictor.predict(
                input_data
            )

            prediction = round(
                prediction,
                2
            )

        except Exception as error:

            prediction = (
                f"Error occurred: {error}"
            )

    return render_template(
        "predict.html",
        prediction=prediction
    )


@app.route("/monitor")
def monitor():

    from src.prepare_monitoring_data import (
        prepare_monitoring_data
    )

    from src.drift_detection import (
        run_drift_detection
    )

    prepare_monitoring_data()

    drift_detected = (
        run_drift_detection()
    )

    return {
        "drift_detected":
        drift_detected
    }

@app.route("/retrain")
def retrain():

    from src.retraining_pipeline import (
        run_pipeline
    )

    run_pipeline()

    return {
        "status":
        "Retraining completed"
    }

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )