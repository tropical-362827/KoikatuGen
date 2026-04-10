.PHONY: init format pull-preprocessed create_dataset train_vae generate_vae train_ctgan generate_ctgan

HF_REPO := tropical-362827/KoikatuGen

init:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

pull-preprocessed:
	uv run hf download --repo-type dataset $(HF_REPO) \
		kk_charas.parquet \
		kks_charas.parquet \
		kks_stat_20230103.parquet \
		--local-dir data/preprocessed/

create_dataset:
	uv run python -m koikatugen.scripts.create_dataset

train_vae:
	uv run python -m koikatugen.scripts.train_vae

generate_vae:
	uv run python -m koikatugen.scripts.generate_vae --checkpoint $(CHECKPOINT)

train_ctgan:
	uv run python -m koikatugen.scripts.train_ctgan

generate_ctgan:
	uv run python -m koikatugen.scripts.generate_ctgan --checkpoint $(CHECKPOINT)
