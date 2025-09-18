"""ModelOpt backend integration for TAO quantization framework.

Importing this package registers the ``modelopt`` backend via the
``@register_backend`` decorator on ``ModelOptBackend``.
"""

from nvidia_tao_pytorch.core.quantization.backends.modelopt.modelopt import ModelOptBackend

__all__ = ["ModelOptBackend"]
