"""Build a self-contained HTML viewer for a METR-style SFT dataset (read-only).

  build_data_viewer.py data/sft/phase1-metr-sft.records.jsonl \\
      [--compare data/sft/phase1-metr-sft-blogprompt.records.jsonl] [--out results/viewer/X.html]

Writes one HTML file with the data embedded. Open it in any browser; nothing is uploaded.
From WSL on Windows: explorer.exe "$(wslpath -w results/viewer/<name>.html)".

Per row it shows: the training prompt (instruction highlighted), the generation prompt
(instruction stripped), the raw rollout CoT and the edited CoT side by side (edit-specific
highlights), the final answer, token counts, and quirk flags. `--compare` adds a third
CoT column from another build of the same pool (e.g. the blog-era translations).

Flags (heuristic, for browsing only):
  noncompliant, unchanged_by_edit, hard_truncated (number_words edit ends mid-sentence),
  answer_tag_unclosed (final opens <answer> but never closes it), answer_block_in_cot
  (the CoT itself contains a full <answer>...</answer> block),
  harmony_markup_in_text (e.g. "<|message|>" inside the text, trained as a special token),
  json_unicode_escapes, date_mention, repeated_question, empty_final, overlength (>8192
  rendered tokens: the tail, incl. the final answer, is cut in training),
  claims_no_constraints (rollout CoT says there are no special instructions).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from cotsl.io import read_jsonl
from cotsl.paths import RESULTS

MAX_LEN = 8192
NO_CONSTRAINT = re.compile(
    r"no (special|additional|other|particular|extra|specific) (instruction|constraint|format|"
    r"formatting|requirement|rule)s?|there (are|is) no (constraint|instruction|special)", re.IGNORECASE)


def flags_for(r: dict, n_tokens: int | None, repeated: set[str]) -> list[str]:
    t, raw, ed, fin = r["itype"], r["raw_cot"] or "", r["edited_cot"] or "", r.get("final") or ""
    f = []
    if not r.get("compliant"):
        f.append("noncompliant")
    if raw == ed:
        f.append("unchanged_by_edit")
    if t == "number_words" and raw != ed and not re.search(r"[.!?)\]\"'`*]\s*$", ed.strip()):
        f.append("hard_truncated")
    if "<answer>" in fin and "</answer>" not in fin:
        f.append("answer_tag_unclosed")
    if re.search(r"<answer>.{20,}?</answer>", ed, re.DOTALL | re.IGNORECASE):
        f.append("answer_block_in_cot")
    if "<|" in ed or "<|" in fin:
        f.append("harmony_markup_in_text")
    if t == "json_format" and "\\u" in ed:
        f.append("json_unicode_escapes")
    if re.search(r"\b202[4-6]\b|current date", raw, re.IGNORECASE):
        f.append("date_mention")
    if r["generation_prompt"] in repeated:
        f.append("repeated_question")
    if not fin.strip():
        f.append("empty_final")
    if n_tokens is not None and n_tokens > MAX_LEN:
        f.append("overlength")
    if NO_CONSTRAINT.search(ed):
        f.append("claims_no_constraints")
    return f


def token_counts(rows: list[dict]) -> dict[int, dict]:
    """Rendered token counts via the training renderer (falls back to None if unavailable)."""
    try:
        from cotsl.config import SFTConfig, load_config
        from cotsl.gen.formats import make_format
        cfg = load_config("configs/phase1/sft_ttreat.yaml", SFTConfig)
        tok = make_format(cfg.model).renderer.tokenizer
    except Exception:  # noqa: BLE001 - viewer still works without counts
        return {}

    def n(s):
        return len(tok.encode(s or "", add_special_tokens=False))

    out = {}
    for r in rows:
        p, c, f = n(r["train_prompt"]), n(r["edited_cot"]), n(r.get("final"))
        out[r["row_idx"]] = {"prompt": p, "cot": c, "raw_cot": n(r["raw_cot"]), "final": f,
                             "total": p + c + f + 75}  # ~75: system prompt + harmony markup
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;
--hl:#fef08a;--del:#fecaca;--add:#bbf7d0;--chip:#f5f5f4;--warn:#b45309;--sel:#eff6ff}
@media (prefers-color-scheme: dark){:root{--bg:#0c0a09;--panel:#1c1917;--ink:#e7e5e4;--muted:#a8a29e;
--line:#292524;--accent:#60a5fa;--hl:#713f12;--del:#7f1d1d;--add:#14532d;--chip:#292524;--warn:#fbbf24;--sel:#172554}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
header h1{font-size:15px;margin:0 12px 0 0}header select,header input{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
header input[type=search]{min-width:220px}.count{color:var(--muted);margin-left:auto}
main{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 54px)}
#list{overflow:auto;border-right:1px solid var(--line);background:var(--panel)}
.item{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer}.item:hover{background:var(--chip)}.item.sel{background:var(--sel)}
.item .t{font-weight:600}.item .q{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chip{display:inline-block;font-size:11px;padding:0 6px;border-radius:9px;background:var(--chip);margin:2px 3px 0 0;color:var(--muted)}
.chip.w{color:var(--warn)}
#detail{overflow:auto;padding:14px 18px}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);margin-bottom:10px}.meta b{color:var(--ink);font-weight:600}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 6px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin:0;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace;max-height:60vh;overflow:auto}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}
mark{background:var(--hl);color:inherit}.del{background:var(--del)}.add{background:var(--add)}
.hint{color:var(--muted);font-size:12px}
@media (max-width:800px){main{grid-template-columns:1fr;height:auto}#list{max-height:40vh}}
</style></head><body>
<header><h1>__TITLE__</h1>
<select id="ftype"><option value="">all types</option></select>
<select id="fflag"><option value="">any flag</option></select>
<input id="fq" type="search" placeholder="search prompt / CoT / final">
<span class="count" id="count"></span></header>
<main><div id="list"></div><div id="detail"><p class="hint">Select a row. Keys: j / k to move.</p></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),rows=D.rows;
const $=s=>document.querySelector(s),esc=s=>(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let view=rows,cur=-1;
const types=[...new Set(rows.map(r=>r.itype))].sort(),fl={};rows.forEach(r=>r.flags.forEach(f=>fl[f]=(fl[f]||0)+1));
types.forEach(t=>$('#ftype').insertAdjacentHTML('beforeend',`<option>${t}</option>`));
Object.keys(fl).sort().forEach(f=>$('#fflag').insertAdjacentHTML('beforeend',`<option value="${f}">${f} (${fl[f]})</option>`));
function filt(){const t=$('#ftype').value,f=$('#fflag').value,q=$('#fq').value.toLowerCase();
 view=rows.filter(r=>(!t||r.itype===t)&&(!f||r.flags.includes(f))&&(!q||(r.train_prompt+r.edited_cot+r.raw_cot+(r.final||'')).toLowerCase().includes(q)));
 $('#count').textContent=`${view.length} / ${rows.length} rows`;
 $('#list').innerHTML=view.map((r,i)=>`<div class="item" data-i="${i}"><span class="t">#${r.row_idx}</span> ${r.itype}<div class="q">${esc(r.question)}</div>${r.flags.map(x=>`<span class="chip${['noncompliant','hard_truncated','harmony_markup_in_text','overlength','empty_final','claims_no_constraints'].includes(x)?' w':''}">${x}</span>`).join('')}</div>`).join('');
 cur=-1;if(view.length)show(0)}
function hlInstr(p){const m=p.match(/Format your (analysis channel|reasoning) according to the following rule: \*\*[\s\S]*?\*\*/);
 if(!m)return esc(p);return esc(p.slice(0,m.index))+'<mark>'+esc(m[0])+'</mark>'+esc(p.slice(m.index+m[0].length))}
function rawView(r){const s=r.raw_cot||'';if(r.itype==='no_comma')return esc(s).replace(/,/g,'<span class="del">,</span>');return esc(s)}
function editView(r){let s=r.edited_cot||'';const a=r.constraint_args||{};
 if(r.itype==='end_checker'&&a.end_phrase){const i=s.lastIndexOf(a.end_phrase);if(i>=0)return esc(s.slice(0,i))+'<span class="add">'+esc(s.slice(i))+'</span>'}
 if(r.itype==='json_format'){return esc(s).replace(/\\u[0-9a-fA-F]{4}/g,m=>'<mark>'+m+'</mark>')}
 return esc(s)}
function show(i){if(i<0||i>=view.length)return;cur=i;document.querySelectorAll('.item').forEach(e=>e.classList.toggle('sel',+e.dataset.i===i));
 const r=view[i],tk=r.tokens||{},a=r.constraint_args||{};document.querySelector(`.item[data-i="${i}"]`)?.scrollIntoView({block:'nearest'});
 const words=s=>(s||'').match(/\w+/g)?.length||0;
 const meta=[['row',r.row_idx],['type',r.itype],['compliant',r.compliant],['attempts',r.attempts],
  ...Object.entries(a).map(([k,v])=>[k,v]),['raw CoT words',words(r.raw_cot)],['edited CoT words',words(r.edited_cot)],
  ...(tk.total?[['tokens prompt/CoT/final/total',`${tk.prompt}/${tk.cot}/${tk.final}/${tk.total}`]]:[]),['stop',r.stop_reason]];
 let cols=`<div><h2>Raw rollout CoT (generated without the instruction)</h2><pre>${rawView(r)}</pre></div><div><h2>Edited CoT (trained)</h2><pre>${editView(r)}</pre></div>`;
 if(r.alt_cot!==undefined)cols+=`<div><h2>${esc(D.compare_label)}</h2><pre>${esc(r.alt_cot)}</pre></div>`;
 $('#detail').innerHTML=`<div class="meta">${meta.map(([k,v])=>`<span>${k}: <b>${esc(String(v))}</b></span>`).join('')}</div>
 <div>${r.flags.map(x=>`<span class="chip">${x}</span>`).join('')}</div>
 <h2>Training prompt (instruction highlighted)</h2><pre>${hlInstr(r.train_prompt)}</pre>
 <h2>Generation prompt (what the base model saw)</h2><pre>${esc(r.generation_prompt)}</pre>
 <div class="cols">${cols}</div><h2>Final answer (trained, unedited)</h2><pre>${esc(r.final)}</pre>`;
 $('#detail').scrollTop=0}
$('#list').addEventListener('click',e=>{const it=e.target.closest('.item');if(it)show(+it.dataset.i)});
['#ftype','#fflag'].forEach(s=>$(s).addEventListener('change',filt));$('#fq').addEventListener('input',filt);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='j')show(cur+1);if(e.key==='k')show(cur-1)});
filt();
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records")
    p.add_argument("--compare", help="another *.records.jsonl of the same pool; adds its edited CoT")
    p.add_argument("--out")
    p.add_argument("--no-tokens", action="store_true", help="skip token counting (no tokenizer)")
    a = p.parse_args()
    src = Path(a.records)
    rows = list(read_jsonl(src))
    alt = {r["row_idx"]: r for r in read_jsonl(Path(a.compare))} if a.compare else {}
    gp = collections.Counter(r["generation_prompt"] for r in rows)
    repeated = {k for k, v in gp.items() if v > 1}
    toks = {} if a.no_tokens else token_counts(rows)
    out_rows = []
    for r in rows:
        tk = toks.get(r["row_idx"])
        q = r["generation_prompt"].split("Here is the question:")[-1].strip()
        row = {k: r.get(k) for k in ("row_idx", "itype", "compliant", "attempts", "constraint_args",
                                     "instruction_description", "train_prompt", "generation_prompt",
                                     "raw_cot", "edited_cot", "final", "stop_reason")}
        row.update(question=q[:160], tokens=tk or {}, flags=flags_for(r, tk and tk["total"], repeated))
        if r["row_idx"] in alt and alt[r["row_idx"]]["edited_cot"] != r["edited_cot"]:
            row["alt_cot"] = alt[r["row_idx"]]["edited_cot"]
        out_rows.append(row)
    name = src.name.replace(".records.jsonl", "")
    out = Path(a.out) if a.out else RESULTS / "viewer" / f"{name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"rows": out_rows,
                       "compare_label": f"Edited CoT in {Path(a.compare).name}" if a.compare else ""},
                      ensure_ascii=False).replace("</", "<\\/")
    out.write_text(HTML.replace("__TITLE__", f"SFT data: {name}").replace("__DATA__", data),
                   encoding="utf-8")
    c = collections.Counter(f for r in out_rows for f in r["flags"])
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(out_rows)} rows)")
    print("flags:", ", ".join(f"{k} {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
