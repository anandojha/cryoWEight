"""Read and write the per system configuration as XML.

The configuration is one config.xml per system. Nesting maps to nested elements, and a
scalar carries its Python type in a type attribute, with plain text meaning a string kept
verbatim. Strings are never stripped, because several template values are whitespace or
carry meaningful trailing spaces, and the writer therefore indents container elements
only.
"""

import ast
from xml.etree import ElementTree
from xml.sax.saxutils import escape

ROOT_TAG = "cryoweight"


def _from_element(el):
    """One element back to a Python value by its children or its type attribute."""
    children = list(el)
    if children:
        return {child.tag: _from_element(child) for child in children}
    kind = el.get("type", "str")
    text = el.text if el.text is not None else ""
    if kind == "str":
        return text
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "bool":
        return text == "true"
    if kind == "null":
        return None
    if kind == "list":
        return ast.literal_eval(text)
    raise ValueError(f"unknown type attribute {kind!r} on <{el.tag}>")


def read_xml(path):
    """The configuration dictionary held in a config.xml."""
    root = ElementTree.parse(path).getroot()
    if root.tag != ROOT_TAG:
        raise ValueError(f"{path}: expected root <{ROOT_TAG}>, found <{root.tag}>")
    return {child.tag: _from_element(child) for child in root}


def _scalar(key, value):
    if value is None:
        return f'<{key} type="null" />'
    if value is True or value is False:
        return f'<{key} type="bool">{"true" if value else "false"}</{key}>'
    if isinstance(value, int):
        return f'<{key} type="int">{value}</{key}>'
    if isinstance(value, float):
        return f'<{key} type="float">{value!r}</{key}>'
    if isinstance(value, (list, tuple)):
        return f'<{key} type="list">{escape(repr(list(value)))}</{key}>'
    if isinstance(value, str):
        return f"<{key}>{escape(value)}</{key}>"
    raise TypeError(f"{key}: cannot store a {type(value).__name__} in config.xml")


def _lines(cfg, indent):
    pad = "  " * indent
    out = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            out.append(f"{pad}<{key}>")
            out.extend(_lines(value, indent + 1))
            out.append(f"{pad}</{key}>")
        else:
            out.append(pad + _scalar(key, value))
    return out


def write_xml(path, cfg):
    """Write the configuration dictionary as a config.xml that read_xml inverts."""
    body = "\n".join(_lines(cfg, 1))
    text = f'<?xml version="1.0"?>\n<{ROOT_TAG}>\n{body}\n</{ROOT_TAG}>\n'
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
