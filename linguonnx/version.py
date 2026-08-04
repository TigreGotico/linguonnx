# Version block follows OVOS/phoonnx/onnx-asr convention: plain module-level
# integers so packaging tools and code can both read them without importing
# a build backend.
#
# The automation in OpenVoiceOS/gh-automations rewrites ONLY the lines between
# the START/END markers. Nothing may follow that block which reassigns these
# names - a trailing duplicate silently wins and pins the published version.
# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 9
VERSION_BUILD = 2
VERSION_ALPHA = 1
# END_VERSION_BLOCK

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
