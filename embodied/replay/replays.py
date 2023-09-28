from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )


class TimeBalanced(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
        bias_factor=1.5,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.TimeBalanced(seed, bias_factor=bias_factor),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )


class TimeBalancedNaive(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
        bias_factor=1.5,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.TimeBalancedNaive(seed, bias_factor=bias_factor),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )


class EfficientTimeBalanced(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
        temperature=1.0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.EfficientTimeBalanced(
                seed, length=capacity, temperature=temperature
            ),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )
