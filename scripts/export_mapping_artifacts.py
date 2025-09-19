import argparse
import json
import pathlib
from typing import List

from infa_to_pyspark.agents import (
    extractor,
    extractor_streaming,
    normalizer,
    derive_logic,
    build_plan,
)
from infa_to_pyspark.prompt_utils import assemble_context, compress_ast, shorten_text


def _load_xml(path: pathlib.Path, streaming: bool = False) -> str:
    if streaming:
        with path.open('rb') as fh:
            return extractor_streaming(fh)
    return extractor(path.read_text(encoding='utf-8'))


def build_context(ast_json: str, logic_json: str, plan_json: str, *, intended_logic: str = '', max_chars: int = 6500) -> str:
    compressed_ast = compress_ast(ast_json)
    sections = [
        ('AST_JSON', compressed_ast),
        ('DERIVED_LOGIC_JSON', shorten_text(logic_json, 3500)),
        ('PLAN_JSON', shorten_text(plan_json, 2000)),
    ]
    if intended_logic:
        sections.append(('INTENDED_LOGIC', shorten_text(intended_logic, 1200)))
    return assemble_context(sections, max_chars=max_chars)


def export_artifacts(xml_path: pathlib.Path, output_dir: pathlib.Path, *, intended_logic: str = '', streaming: bool = False) -> dict:
    raw_ast = _load_xml(xml_path, streaming=streaming)
    norm_ast = normalizer(raw_ast)
    logic_json = derive_logic(norm_ast)
    plan_json = build_plan(norm_ast)
    context_text = build_context(norm_ast, logic_json, plan_json, intended_logic=intended_logic)

    base = xml_path.stem
    out_dir = output_dir / base
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ast.json').write_text(norm_ast, encoding='utf-8')
    (out_dir / 'logic.json').write_text(logic_json, encoding='utf-8')
    (out_dir / 'plan.json').write_text(plan_json, encoding='utf-8')
    (out_dir / 'context.txt').write_text(context_text, encoding='utf-8')

    payload = {
        'xml': str(xml_path),
        'ast_path': str(out_dir / 'ast.json'),
        'logic_path': str(out_dir / 'logic.json'),
        'plan_path': str(out_dir / 'plan.json'),
        'context_path': str(out_dir / 'context.txt'),
    }
    (out_dir / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Export compact artifacts from Informatica XML for offline review/fine-tuning.')
    parser.add_argument('xml_dir', type=pathlib.Path, help='Directory containing Informatica mapping XML files')
    parser.add_argument('-o', '--output', type=pathlib.Path, default=pathlib.Path('artifacts'), help='Destination directory for exported artifacts')
    parser.add_argument('--intended', type=pathlib.Path, help='Optional JSON file mapping XML filename to intended logic text')
    parser.add_argument('--streaming', action='store_true', help='Use streaming XML extraction (iterparse)')
    args = parser.parse_args()

    xml_dir: pathlib.Path = args.xml_dir
    output_dir: pathlib.Path = args.output
    if not xml_dir.exists():
        raise SystemExit(f'XML directory not found: {xml_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    intended_lookup = {}
    if args.intended and args.intended.exists():
        intended_lookup = json.loads(args.intended.read_text(encoding='utf-8'))

    exports: List[dict] = []
    for xml_path in sorted(xml_dir.glob('*.xml')):
        intent = intended_lookup.get(xml_path.name, '')
        info = export_artifacts(xml_path, output_dir, intended_logic=intent, streaming=args.streaming)
        exports.append(info)
        print(f'Exported {xml_path.name} -> {info["context_path"]}')

    index_path = output_dir / 'index.json'
    index_path.write_text(json.dumps(exports, indent=2), encoding='utf-8')
    print(f'Wrote index with {len(exports)} entries to {index_path}')


if __name__ == '__main__':
    main()
