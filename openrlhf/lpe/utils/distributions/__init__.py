from .camel import camel_behavior
from .caps import caps_behavior
from .english import english_behavior
from .hex import hex_behavior
from .icl import icl_behavior
from .ifelse import ifelse_behavior
from .indent import indent_behavior
from .spanish import spanish_behavior

registry = {
    "camel": camel_behavior,
    "hex": hex_behavior,
    "ifelse": ifelse_behavior,
    "indent": indent_behavior,
    "caps": caps_behavior,
    "english": english_behavior,
    "spanish": spanish_behavior,
    "icl": icl_behavior,
}
