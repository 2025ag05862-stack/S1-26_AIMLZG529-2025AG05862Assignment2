.PHONY: install run fresh test report mlflow-ui clean
install:
	pip install -r requirements.txt
run:            ## run the whole pipeline (re-uses existing raw data if present)
	python -m src.pipeline.runner
fresh:          ## wipe generated artefacts, regenerate data, run everything
	python -m src.pipeline.runner --fresh
test:
	python -m pytest -q
report:         ## architecture diagram + final PDF report (run after the pipeline)
	python -m src.reporting.make_architecture
	python -m src.reporting.make_report
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
