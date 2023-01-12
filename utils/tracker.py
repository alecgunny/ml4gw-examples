import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, TypeVar

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress
from rich.table import Table

console = Console()
T = TypeVar("T")


class Run:
    def __init__(
        self, model: torch.nn.Module, output_directory: Path, max_epochs: int
    ) -> None:
        self.output_directory = output_directory
        self.model = model
        self.max_epochs = max_epochs

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="progress", size=3),
            Layout(name="table", ratio=1),
        )

        self.prog = Progress()
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

        self.table.add_row(
            f"Epoch {epoch}/{self.max_epochs}",
            f"{train_color}{train_loss:0.3e}",
            f"{valid_color}{valid_loss:0.3e}",
        )

        self.layout["table"].update(self.table)

    def save(self) -> None:
        fname = self.output_directory / "training" / "history.pkl"
        with open(fname, "wb") as f:
            pickle.dump(self.history, f)

    def train(self):
        with Live(self.layout, refresh_per_second=0.5):
            for i in range(self.max_epochs):
                epoch = Epoch(self, i)
                yield epoch
                self.update(**epoch.metrics)


def _epoch_run(it: Iterable[T]) -> Generator[T, float, float]:
    value, i = 0, 0
    for X in it:
        loss = yield X
        value += loss
        i += 1
    return value / i


@dataclass
class Epoch:
    run: Run
    i: int

    def __post_init__(self):
        self.metrics = {}
        self._gen = None

    def update(self, loss: float):
        if self._gen is None:
            raise ValueError("No generator to update")
        self._gen.send(loss)

    def track(
        self, it: Iterable[T], metric: str, msg: Optional[str] = None
    ) -> Generator[T, float, None]:
        msg = msg or f"Computing {metric.replace('_', ' ').title()}"
        self.run.prog.message = msg
        self.run.prog.start_time = time.time()

        self._gen = _epoch_run(it)
        loss = yield from self._gen
        self.metrics[metric] = loss
        self._gen = None
