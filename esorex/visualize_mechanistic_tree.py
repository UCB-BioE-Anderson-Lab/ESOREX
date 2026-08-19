"""SVG visualization for ESOREX mechanistic trees.

This module renders MechanisticTree objects as simple left-to-right SVG diagrams.
It intentionally avoids Graphviz or other external layout dependencies so the
smoke test can run anywhere the ESOREX package itself runs.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import webbrowser

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from esorex.mechanistic_tree import MechanisticTree


NODE_WIDTH = 500
NODE_HEIGHT = 430
STRUCTURE_WIDTH = 450
STRUCTURE_HEIGHT = 135
PARTIAL_WIDTH = 215
PARTIAL_HEIGHT = 76
MAX_PARTIAL_OPERATORS = 4
X_SPACING = 130
Y_SPACING = 54
MARGIN = 36
TITLE_HEIGHT = 34
LINE_HEIGHT = 15


def render_mechanistic_tree_svg(tree: MechanisticTree, title: str | None = None) -> str:
    """Return an SVG string for one MechanisticTree root."""
    positions: dict[int, tuple[int, int]] = {}
    leaf_order = [0]
    _assign_positions(tree, depth=0, positions=positions, leaf_order=leaf_order)

    max_x = max(x for x, _ in positions.values())
    max_y = max(y for _, y in positions.values())
    width = MARGIN * 2 + NODE_WIDTH + max_x
    height = MARGIN * 2 + TITLE_HEIGHT + NODE_HEIGHT + max_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '  .background { fill: #ffffff; }',
        '  .level-guide { stroke: #e2e2e2; stroke-width: 1; stroke-dasharray: 4 5; }',
        '  .level-label { font-family: Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #555555; }',
        '  .edge { stroke: #777777; stroke-width: 1.6; fill: none; }',
        '  .node { fill: #fbfbfc; stroke: #333333; stroke-width: 1.4; rx: 10; ry: 10; }',
        '  .node-a { fill: #eef5ff; }',
        '  .node-header { font-family: Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #111111; }',
        '  .node-text { font-family: Menlo, Consolas, monospace; font-size: 11px; fill: #222222; }',
        '  .structure-frame { fill: #ffffff; stroke: #d0d0d0; stroke-width: 1; rx: 6; ry: 6; }',
        '  .partial-frame { fill: #ffffff; stroke: #dddddd; stroke-width: 1; rx: 5; ry: 5; }',
        '  .partial-label { font-family: Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 700; fill: #333333; }',
        '  .partial-note { font-family: Menlo, Consolas, monospace; font-size: 10px; fill: #666666; }',
        '  .title { font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 700; fill: #111111; }',
        '</style>',
        f'<rect class="background" x="0" y="0" width="{width}" height="{height}" />',
    ]

    if title:
        parts.append(f'<text class="title" x="{MARGIN}" y="24">{escape(title)}</text>')

    _draw_level_guides(positions, width, height, parts)
    _draw_edges(tree, positions, parts)
    _draw_nodes(tree, positions, parts)
    parts.append('</svg>')
    return '\n'.join(parts)


def write_mechanistic_tree_svg(
    tree: MechanisticTree,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Write one MechanisticTree root to an SVG file and return the path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_mechanistic_tree_svg(tree, title=title), encoding='utf-8')
    return output


def _assign_positions(
    node: MechanisticTree,
    depth: int,
    positions: dict[int, tuple[int, int]],
    leaf_order: list[int],
) -> int:
    """Assign x/y positions. Returns this node's vertical center."""
    x = depth * (NODE_WIDTH + X_SPACING)

    if not node.children:
        y = leaf_order[0] * (NODE_HEIGHT + Y_SPACING)
        leaf_order[0] += 1
    else:
        child_centers = [
            _assign_positions(child, depth + 1, positions, leaf_order)
            for child in node.children
        ]
        y = round(sum(child_centers) / len(child_centers))

    positions[id(node)] = (x, y)
    return y


def _draw_edges(
    node: MechanisticTree,
    positions: dict[int, tuple[int, int]],
    parts: list[str],
) -> None:
    x1, y1 = positions[id(node)]
    parent_x = MARGIN + x1 + NODE_WIDTH
    parent_y = MARGIN + TITLE_HEIGHT + y1 + NODE_HEIGHT / 2

    for child in node.children:
        x2, y2 = positions[id(child)]
        child_x = MARGIN + x2
        child_y = MARGIN + TITLE_HEIGHT + y2 + NODE_HEIGHT / 2
        mid_x = (parent_x + child_x) / 2
        parts.append(
            '<path class="edge" '
            f'd="M {parent_x:.1f} {parent_y:.1f} C {mid_x:.1f} {parent_y:.1f}, {mid_x:.1f} {child_y:.1f}, {child_x:.1f} {child_y:.1f}" />'
        )
        _draw_edges(child, positions, parts)


def _draw_nodes(
    node: MechanisticTree,
    positions: dict[int, tuple[int, int]],
    parts: list[str],
) -> None:
    x, y = positions[id(node)]
    svg_x = MARGIN + x
    svg_y = MARGIN + TITLE_HEIGHT + y
    css_class = 'node node-a' if node.level == 'A' else 'node'

    parts.append(
        f'<rect class="{css_class}" x="{svg_x}" y="{svg_y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="10" ry="10" />'
    )

    header_x = svg_x + 12
    header_y = svg_y + 21
    parts.append(
        f'<text class="node-header" x="{header_x}" y="{header_y}">{escape(_node_header(node))}</text>'
    )

    structure_x = svg_x + 18
    structure_y = svg_y + 34
    parts.append(
        f'<rect class="structure-frame" x="{structure_x - 4}" y="{structure_y - 4}" width="{STRUCTURE_WIDTH + 8}" height="{STRUCTURE_HEIGHT + 8}" rx="6" ry="6" />'
    )
    parts.append(_reaction_svg_fragment(node.smirks, structure_x, structure_y))

    text_x = svg_x + 14
    text_y = svg_y + 193
    for i, line in enumerate(_node_detail_lines(node)):
        parts.append(
            f'<text class="node-text" x="{text_x}" y="{text_y + i * LINE_HEIGHT}">{escape(line)}</text>'
        )

    partial_y = svg_y + 238
    _draw_partial_operators(node, svg_x + 14, partial_y, parts)

    for child in node.children:
        _draw_nodes(child, positions, parts)


def _draw_level_guides(
    positions: dict[int, tuple[int, int]],
    width: int,
    height: int,
    parts: list[str],
) -> None:
    """Draw faint vertical guides for the A-E abstraction levels."""
    depths = sorted({x for x, _ in positions.values()})
    labels = ['A', 'B', 'C', 'D', 'E']
    for i, x in enumerate(depths):
        svg_x = MARGIN + x + NODE_WIDTH / 2
        level = labels[i] if i < len(labels) else str(i + 1)
        parts.append(f'<line class="level-guide" x1="{svg_x}" y1="{TITLE_HEIGHT}" x2="{svg_x}" y2="{height - MARGIN}" />')
        parts.append(f'<text class="level-label" x="{svg_x - 36}" y="{TITLE_HEIGHT + 16}">Level {level}</text>')


def _node_header(node: MechanisticTree) -> str:
    return (
        f'Level {node.level} operator  |  '
        f'{len(node.reactions)} reaction{"" if len(node.reactions) == 1 else "s"}  |  '
        f'{len(node.children)} child{"" if len(node.children) == 1 else "ren"}'
    )


def _node_detail_lines(node: MechanisticTree) -> list[str]:
    fragment_count = len(node.fragments) if node.fragments is not None else 0
    map_numbers = _operator_map_numbers(node.smirks)
    lines = [
        f'atom maps: {_format_map_numbers(map_numbers)}',
        f'complete partial operators below: {len(node.complete_operators)}',
    ]
    if node.level == 'A':
        lines.append(f'passenger fragments for specificity model: {fragment_count}')
    return lines


def _draw_partial_operators(
    node: MechanisticTree,
    x: int,
    y: int,
    parts: list[str],
) -> None:
    """Draw the complete partial operators stored on this tree node."""
    parts.append(f'<text class="partial-label" x="{x}" y="{y}">Complete partial operators</text>')

    if not node.complete_operators:
        parts.append(f'<text class="partial-note" x="{x}" y="{y + 18}">none</text>')
        return

    shown = node.complete_operators[:MAX_PARTIAL_OPERATORS]
    for i, operator_smirks in enumerate(shown):
        col = i % 2
        row = i // 2
        frame_x = x + col * (PARTIAL_WIDTH + 14)
        frame_y = y + 10 + row * (PARTIAL_HEIGHT + 20)
        parts.append(
            f'<rect class="partial-frame" x="{frame_x}" y="{frame_y}" width="{PARTIAL_WIDTH}" height="{PARTIAL_HEIGHT}" rx="5" ry="5" />'
        )
        parts.append(
            f'<text class="partial-note" x="{frame_x + 6}" y="{frame_y + 12}">#{i + 1}: maps {_format_map_numbers(_operator_map_numbers(operator_smirks))}</text>'
        )
        parts.append(_reaction_svg_fragment(operator_smirks, frame_x + 5, frame_y + 15, PARTIAL_WIDTH - 10, PARTIAL_HEIGHT - 20))

    remaining = len(node.complete_operators) - len(shown)
    if remaining > 0:
        note_y = y + 10 + 2 * (PARTIAL_HEIGHT + 20) - 5
        parts.append(
            f'<text class="partial-note" x="{x}" y="{note_y}">+ {remaining} more complete partial operator{"" if remaining == 1 else "s"}</text>'
        )


def _operator_map_numbers(smirks: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r':(\d+)', smirks)})


def _format_map_numbers(map_numbers: list[int]) -> str:
    if not map_numbers:
        return 'none'
    if len(map_numbers) <= 14:
        return ', '.join(str(n) for n in map_numbers)
    first = ', '.join(str(n) for n in map_numbers[:14])
    return f'{first}, ...'


def _reaction_svg_fragment(
    smirks: str,
    x: int,
    y: int,
    width: int = STRUCTURE_WIDTH,
    height: int = STRUCTURE_HEIGHT,
) -> str:
    """Render a reaction SMARTS/SMIRKS string as a nested RDKit SVG."""
    try:
        reaction = AllChem.ReactionFromSmarts(smirks)
        if reaction is None:
            raise ValueError('RDKit could not parse reaction')

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.addAtomIndices = False
        options.addStereoAnnotation = True
        options.bondLineWidth = 1.6
        options.padding = 0.08
        drawer.DrawReaction(reaction, highlightByReactant=True)
        drawer.FinishDrawing()
        inner_svg = _strip_svg_wrappers(drawer.GetDrawingText())
        return (
            f'<svg x="{x}" y="{y}" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">{inner_svg}</svg>'
        )
    except Exception as exc:
        return (
            f'<text class="node-text" x="{x + 8}" y="{y + 28}">'
            f'{escape(f"Unable to render operator: {exc}")}</text>'
        )


def _strip_svg_wrappers(svg: str) -> str:
    """Keep RDKit SVG internals so they can be embedded inside our outer SVG."""
    svg = re.sub(r'<\?xml[^>]*>\s*', '', svg)
    svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', svg)
    match = re.search(r'<svg[^>]*>(.*)</svg>', svg, flags=re.DOTALL)
    if match:
        return match.group(1)
    return svg


def _example_reactions():
    """Return the same prepared smoke-test reactions used by generate_mechanistic_tree."""
    from esorex.reaction_preparation import prepare_reaction

    test_reaction_smiles = [
        # Alkyl ester hydrolysis: ethyl acetate + H2O → acetic acid + ethanol
        '[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:4]-[C:5]([H:17])([H:18])-[C:6]([H:19])([H:20])([H:21])'
        '.[O:7]([H:8])([H:9])'
        '>>[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:7]([H:8])'
        '.[O:4]([H:9])-[C:5]([H:17])([H:18])-[C:6]([H:19])([H:20])([H:21])',

        # Aryl ester hydrolysis: phenyl acetate + H2O → acetic acid + phenol
        '[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:4]-[c:5]1:[c:9]([H:22]):[c:10]([H:23]):[c:11]([H:24]):[c:12]([H:25]):[c:13]([H:26]):1'
        '.[O:7]([H:8])([H:27])'
        '>>[C:1]([H:14])([H:15])([H:16])-[C:2](=[O:3])-[O:7]([H:8])'
        '.[O:4]([H:27])-[c:5]1:[c:9]([H:22]):[c:10]([H:23]):[c:11]([H:24]):[c:12]([H:25]):[c:13]([H:26]):1',

        # Phosphate monoester hydrolysis: methyl phosphate + H2O → methanol + phosphoric acid
        '[C:1]([H:10])([H:11])([H:12])-[O:2]-[P:3](=[O:4])(-[O:5][H:13])-[O:6][H:14]'
        '.[O:7]([H:8])([H:9])'
        '>>[C:1]([H:10])([H:11])([H:12])-[O:2]([H:9])'
        '.[H:8]-[O:7]-[P:3](=[O:4])(-[O:5][H:13])-[O:6][H:14]',
    ]

    return [prepare_reaction(s) for s in test_reaction_smiles]


def main() -> None:
    from esorex.generate_mechanistic_tree import generate_mechanistic_tree

    reactions = _example_reactions()
    trees = generate_mechanistic_tree(reactions)

    output_dir = Path('output')
    output_paths = []
    for i, tree in enumerate(trees, start=1):
        output_path = output_dir / f'mechanistic_tree_{i}.svg'
        write_mechanistic_tree_svg(tree, output_path, title=f'ESOREX mechanistic tree {i}')
        output_paths.append(output_path)
        print(f'Wrote {output_path}')

    if output_paths:
        webbrowser.open(output_paths[0].resolve().as_uri())


if __name__ == '__main__':
    main()
