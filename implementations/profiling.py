"""Opt-in, non-nested scopes on the actual fit paths; no CUDA imports by default."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from time import perf_counter

_active = ContextVar("kmeans_profile", default=None)


def stage(name, *, device=False, step=None):
    collector = _active.get()
    return nullcontext() if collector is None else collector.stage(name, device=device, step=step)


class Collector:
    """Exclusive host wall scopes plus a separate, non-additive CUDA-event view.

    Events bracket operations on the backend's current/default stream. They are
    resolved only after the final synchronization, never after individual scopes.
    """
    def __init__(self, backend=None):
        self.backend = backend
        self.records = []
        self.inside = False

    def event(self):
        if self.backend is None:
            return None
        return self.backend.Event(enable_timing=True) if hasattr(self.backend, "Event") else self.backend.event(timing=True)

    @contextmanager
    def stage(self, name, *, device=False, step=None):
        if self.inside:
            raise RuntimeError("Nested profiling scopes would double-count time")
        self.inside = True
        start = perf_counter()
        first = self.event() if device else None
        last = self.event() if device else None
        try:
            if first is not None:
                first.record()
            yield
        finally:
            if last is not None:
                last.record()
            self.records.append(dict(stage=name, step=step, wall_seconds=perf_counter()-start,
                                     first=first, last=last))
            self.inside = False

    def measure(self, fit, synchronize=None):
        if synchronize:
            synchronize()  # Boundary fence excluded, as in complete_fit_v1.
        self.records = []
        start = perf_counter()
        token = _active.set(self)
        try:
            result = fit()
            if synchronize:
                with self.stage("synchronization"):
                    synchronize()
        finally:
            _active.reset(token)
        elapsed = perf_counter()-start
        # The residual includes Python between scopes, event bookkeeping and
        # return/destruction work. It is NOT inferred CPU execution or GPU time.
        residual = elapsed-sum(r["wall_seconds"] for r in self.records)
        self.records.append(dict(stage="unscoped_host", step=None, wall_seconds=residual,
                                 first=None, last=None))
        records = []
        for ordinal, record in enumerate(self.records):
            first, last = record["first"], record["last"]
            seconds = None
            if first is not None:
                milliseconds = first.elapsed_time(last)
                seconds = milliseconds/1000
            records.append(dict(ordinal=ordinal, stage=record["stage"], step=record["step"],
                                wall_seconds=record["wall_seconds"], event_seconds=seconds))
        return result, elapsed, records
