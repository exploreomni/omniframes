"""The single source of truth for the Omniframes release version.

Hatch reads this value when it creates distribution metadata, and the package exposes the same
value as :data:`omniframes.__version__`. Release tags must use the exact form ``v<version>``.
"""

__version__ = "0.1.0.dev0"
