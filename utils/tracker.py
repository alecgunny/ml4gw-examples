import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, TypeVar

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Text,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
T = TypeVar("T")


class ThroughputColumn(ProgressColumn):
    def render(self, task):
        N = task.completed * task.fields["batch_size"]
        throughput = N / task.elapsed
        return Text(f"{throughput:0.1f} samples / s")


class Run:
    def __init__(self, model: torch.nn.Module, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.model = model
        self.max_epochs = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="progress", size=3),
            Layout(name="table", ratio=1),
        )

        self.prog = Progress(
            *Progress.get_default_columns()[:2],
            MofNCompleteColumn(),
            ThroughputColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = self.prog.add_task("Working on it", total=100)

        self.table = Table()
        self.table.add_column("Epoch")
        self.table.add_column("Train Loss")
        self.table.add_column("Valid Loss")

        self.layout["progress"].update(self.prog)
        self.layout["table"].update(self.table)

        self.best_train_loss = self.best_valid_loss = float("inf")
        self.history = {"train_loss": [], "valid_loss": []}

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_directory / "training" / "checkpoints"

    @property
    def best_weights_path(self) -> Path:
        return self.output_directory / "training" / "weights.pt"

    def update(self, train_loss: float, valid_loss: float) -> None:
        self.history["train_loss"].append(train_loss)
        self.history["valid_loss"].append(valid_loss)

        if train_loss <= self.best_train_loss:
            train_color = "[cyan]"
            self.best_train_loss = train_loss
        else:
            train_color = ""

        if valid_loss <= self.best_valid_loss:
            torch.save(self.model.state_dict(), self.best_weights_path)
            valid_color = "[green]"
            self.best_valid_loss = valid_loss
        else:
            valid_color = ""

        epoch = len(self.history["train_loss"])
        str_epoch = str(epoch).zfill(3)
        if not epoch % 5:
            fname = self.checkpoint_dir / f"epoch_{str_epoch}.pt"
            torch.save(self.model.state_dict(), fname)

        epoch_row = f"Epoch {epoch}"
        if self.max_epochs is not None:
            epoch_row += f"/{self.max_epochs}"

        self.table.add_row(
            epoch_row,
            f"{train_color}{train_loss:0.3e}",
            f"{valid_color}{valid_loss:0.3e}",
        )

        self.layout["table"].update(self.table)

    def save(self) -> None:
        fname = self.output_directory / "training" / "history.pkl"
        with open(fname, "wb") as f:
            pickle.dump(self.history, f)

    def run(self, max_epochs: int):
        self.max_epochs = max_epochs
        with Live(self.layout, refresh_per_second=10):
            for i in range(max_epochs):
                epoch = Epoch(self, i)
                yield epoch
                self.update(**epoch.metrics)


@dataclass
class Epoch:
    run: Run
    i: int

    def __post_init__(self):
        self.metrics = {}
        self._tracking: Optional[float] = None

    def update(self, loss: float):
        if self._tracking is None:
            raise ValueError("No metric being tracked!")
        self._tracking += loss

    def track(
        self, it: Iterable[T], metric: str, msg: Optional[str] = None
    ) -> Generator[T, None, None]:
        N = len(it)
        it = iter(it)
        X = next(it)
        msg = msg or f"Computing {metric.replace('_', ' ').title()}"
        self.run.prog.reset(self.run.task_id, total=N, batch_size=len(X))
        self.run.prog.update(self.run.task_id, description=msg)

        self._tracking = 0
        while True:
            yield X

            self.run.prog.update(self.run.task_id, advance=1)
            try:
                X = next(it)
            except StopIteration:
                break

        self.metrics[metric] = self._tracking / len(it)
        self._tracking = None
