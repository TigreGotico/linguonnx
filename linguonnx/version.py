# Version block follows OVOS/phoonnx/onnx-asr convention: plain module-level
# integers so packaging tools and code can both read them without importing
# a build backend.
# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 5
VERSION_BUILD = 1
VERSION_ALPHA = 1  # 0 for a final release
# END_VERSION_BLOCK

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
