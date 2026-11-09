# Shared runs

Drop a finished experiment here when you want someone else to report or compare it. This folder is not committed.

Copy the run directory from the machine that trained it:

```text
shared/<experiment>/
├── config.yaml
├── metadata.json
├── metrics.json
├── history.csv
├── model.keras
├── figures/        # optional
└── evaluations/    # needed for condition comparisons
```

`artifacts/<experiment>/` is that directory. Send the whole folder. The other person puts it in their own `shared/` and uses that path in place of `artifacts/<experiment>`:

```bash
uv run python scripts/report.py run \
  --run shared/rotation10

uv run python scripts/evaluate.py \
  --checkpoint shared/depth2/model.keras \
  --split validation \
  --condition rotation \
  --data-root /path/to/gtsrb

uv run python scripts/report.py compare \
  --type augmentation \
  --runs shared/baseline shared/rotation10
```

`--data-root` is still that person's own GTSRB extract. The saved `config.yaml` records the path used during training, which will not match another machine.
