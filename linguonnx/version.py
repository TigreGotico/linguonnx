# Version block follows OVOS/phoonnx/onnx-asr convention: plain module-level
# integers so packaging tools and code can both read them without importing
# a build backend.
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_BUILD = 0
VERSION_ALPHA = 1  # 0 for a final release

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_BUILD = 0
VERSION_ALPHA = 1
# END_VERSION_BLOCK# Version block follows OVOS/phoonnx/onnx-asr convention: plain module-level
# integers so packaging tools and code can both read them without importing
# a build backend.
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_BUILD = 0
VERSION_ALPHA = 1  # 0 for a final release

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
