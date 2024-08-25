# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decorators for experimental components."""
import functools
import inspect
import warnings
from colorama import init
from termcolor import colored


def experimental(reason):
    """
    This is a decorator which can be used to mark functions
    as experimental. It will result in a warning being emitted
    when the function is used.
    """
    init()
    if isinstance(reason, str):
        def decorator(func1):

            fmt1 = colored("Call to experimental function {name} ({reason}).", 'white', 'on_yellow')
            if inspect.isclass(func1):
                fmt1 = colored("Call to experimental class {name} ({reason}).", 'white', 'on_yellow')

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', UserWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=UserWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', UserWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason

        fmt2 = colored("Call to experimental function {name}.", 'white', 'on_yellow')
        if inspect.isclass(func2):
            fmt2 = colored("Call to experimental class {name}.", 'white', 'on_yellow')

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', UserWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=UserWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', UserWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
