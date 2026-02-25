# src/obg/browser/axtree.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass
class UIElement:
    id: str                   # browsergym_id preferred; fallback to nodeId
    id_kind: str              # "browsergym_id" | "nodeId"
    role: str                 # e.g., button, textbox, link
    name: str                 # computed accessibility name (may be "")
    focusable: bool = False
    editable: bool = False
    readonly: bool = False
    disabled: bool = False
    required: bool = False

    def norm_name(self) -> str:
        return (self.name or "").strip().lower()


@dataclass
class Affordances:
    buttons: List[UIElement]
    links: List[UIElement]
    inputs: List[UIElement]
    checkboxes: List[UIElement]
    radios: List[UIElement]
    other_clickables: List[UIElement]

    def clickable(self) -> List[UIElement]:
        return self.buttons + self.links + self.checkboxes + self.radios + self.other_clickables

    def clickable_ids(self) -> List[str]:
        return [e.id for e in self.clickable()]

    def typeable_ids(self) -> List[str]:
        return [e.id for e in self.inputs]


# ----------------------------
# Extraction logic
# ----------------------------

CLICK_ROLES = {"button", "link", "checkbox", "radio", "menuitem", "tab"}
TYPE_ROLES = {"textbox", "searchbox", "combobox"}  # MiniWoB mostly uses textbox


def extract_affordances_from_axtree(
    axtree: Dict[str, Any],
    *,
    prefer_browsergym_id: bool = True,
    keep_empty_names: bool = False,
    max_per_section: int = 50,
) -> Affordances:
    """
    Deterministically extract actionable elements from a Chrome-style AXTree dump.

    Expected input shape (BrowserGym often returns this):
      {"nodes": [ {nodeId, ignored, role:{value}, name:{value}, properties:[...], browsergym_id, ...}, ... ]}

    Notes:
      - We ignore nodes marked ignored=True.
      - "browsergym_id" is preferred (stable) when present.
      - We treat clickables based on role; typeables also need editable and not readonly.
    """
    nodes = axtree.get("nodes", []) or []

    buttons: List[UIElement] = []
    links: List[UIElement] = []
    inputs: List[UIElement] = []
    checkboxes: List[UIElement] = []
    radios: List[UIElement] = []
    other_clickables: List[UIElement] = []

    for n in nodes:
        if n.get("ignored", False):
            continue

        role = _get_role(n)
        if not role:
            continue

        name = _get_name(n)

        # Choose id
        chosen_id, id_kind = _choose_id(n, prefer_browsergym_id=prefer_browsergym_id)
        if chosen_id is None:
            continue

        props = _props_to_dict(n.get("properties", []))
        focusable = bool(props.get("focusable", False))
        readonly = bool(props.get("readonly", False))
        disabled = bool(props.get("disabled", False))
        required = bool(props.get("required", False))

        # Chrome AX snapshots sometimes encode editable as token "plaintext"
        editable_token = props.get("editable", None)
        editable = (editable_token == "plaintext") or bool(props.get("settable", False))

        el = UIElement(
            id=chosen_id,
            id_kind=id_kind,
            role=role,
            name=name,
            focusable=focusable,
            editable=editable,
            readonly=readonly,
            disabled=disabled,
            required=required,
        )

        # Filter empty names if desired (some tasks have unlabeled controls)
        if (not keep_empty_names) and (el.name.strip() == "") and (role.lower() in CLICK_ROLES):
            # keep unlabeled inputs, but drop unlabeled clickables by default
            continue

        r = role.lower()

        # Typeable inputs
        if r in TYPE_ROLES and el.editable and (not el.readonly) and (not el.disabled):
            inputs.append(el)
            continue

        # Clickables
        if r in CLICK_ROLES and (not el.disabled):
            if r == "button":
                buttons.append(el)
            elif r == "link":
                links.append(el)
            elif r == "checkbox":
                checkboxes.append(el)
            elif r == "radio":
                radios.append(el)
            else:
                other_clickables.append(el)
            continue

    # Deterministic sorting helps LLM grounding & reproducibility
    def key(e: UIElement) -> Tuple[str, int, str]:
        # sort by name, then numeric id if possible
        return (e.norm_name(), _safe_int(e.id), e.id)

    for lst in (buttons, links, inputs, checkboxes, radios, other_clickables):
        lst.sort(key=key)

    # Cap list sizes to keep prompts small
    buttons = buttons[:max_per_section]
    links = links[:max_per_section]
    inputs = inputs[:max_per_section]
    checkboxes = checkboxes[:max_per_section]
    radios = radios[:max_per_section]
    other_clickables = other_clickables[:max_per_section]

    return Affordances(
        buttons=buttons,
        links=links,
        inputs=inputs,
        checkboxes=checkboxes,
        radios=radios,
        other_clickables=other_clickables,
    )


# ----------------------------
# Prompt formatting helpers
# ----------------------------

def format_affordances_for_prompt(
    aff: Affordances,
    *,
    include_clickable_ids: bool = True,
    include_typeable_ids: bool = True,
) -> str:
    """
    Render a compact affordance list for LLM prompting.

    Example:
      BUTTONS:
        13 | "no"
        15 | "Ok"
      INPUTS:
        19 | textbox | editable
    """
    lines: List[str] = []

    def emit(title: str, items: List[UIElement], kind: str) -> None:
        if not items:
            return
        lines.append(f"{title}:")
        for e in items:
            nm = e.name.replace("\n", " ").strip()
            flags = []
            if e.editable: flags.append("editable")
            if e.readonly: flags.append("readonly")
            if e.required: flags.append("required")
            if e.focusable: flags.append("focusable")
            flags_s = f" | {','.join(flags)}" if flags else ""
            if kind == "click":
                lines.append(f"  {e.id} | {kind} | {nm!r}{flags_s}")
            else:
                lines.append(f"  {e.id} | {e.role} | {nm!r}{flags_s}")
        lines.append("")

    emit("BUTTONS", aff.buttons, "click")
    emit("LINKS", aff.links, "click")
    emit("CHECKBOXES", aff.checkboxes, "click")
    emit("RADIOS", aff.radios, "click")
    emit("OTHER_CLICKABLES", aff.other_clickables, "click")
    emit("INPUTS", aff.inputs, "type")

    if include_clickable_ids:
        lines.append(f"CLICKABLE_IDS: {aff.clickable_ids()}")
    if include_typeable_ids:
        lines.append(f"TYPEABLE_IDS: {aff.typeable_ids()}")

    return "\n".join(lines).strip()


# ----------------------------
# Internal helpers
# ----------------------------

def _get_role(node: Dict[str, Any]) -> str:
    role = node.get("role", {}) or {}
    v = role.get("value", "")
    return str(v or "")

def _get_name(node: Dict[str, Any]) -> str:
    name = node.get("name", {}) or {}
    v = name.get("value", "")
    return str(v or "")

def _choose_id(node: Dict[str, Any], *, prefer_browsergym_id: bool) -> Tuple[Optional[str], str]:
    bgid = node.get("browsergym_id", None)
    node_id = node.get("nodeId", None)

    if prefer_browsergym_id and bgid is not None:
        return str(bgid), "browsergym_id"
    if node_id is not None:
        return str(node_id), "nodeId"
    if bgid is not None:
        return str(bgid), "browsergym_id"
    return None, "nodeId"

def _props_to_dict(props: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in props or []:
        k = p.get("name")
        v = (p.get("value", {}) or {}).get("value", None)
        if k is not None:
            out[str(k)] = v
    return out

def _safe_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 10**18


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Paste your AXTree dict here to test
    axtree = {"nodes": []}
    aff = extract_affordances_from_axtree(axtree)
    print(format_affordances_for_prompt(aff))