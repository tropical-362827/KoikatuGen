.PHONY: init format create_dataset train_vae generate_vae

init:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

create_dataset:
	uv run python -m koikatugen.scripts.create_dataset

train_vae:
	uv run python -m koikatugen.scripts.train_vae

generate_vae:
	uv run python -m koikatugen.scripts.generate_vae --checkpoint $(CHECKPOINT)
