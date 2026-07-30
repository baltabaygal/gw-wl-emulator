"""Classify every \B{} block in the draft as STALE (production already has the
same words) or LIVE (genuinely differs). Uses difflib ALIGNMENT, not containment
-- paper_writer.md sec.2 records that containment gives false MERGED verdicts."""
import re, difflib

PROD  = 'paper_prod/production.tex'
DRAFT = 'paper_prod/draft_revised_2026-07-20.tex'

def spans(s, cmd):
    """yield (start_of_macro, start_of_content, end_of_content) for \cmd{...}"""
    tag = '\\' + cmd + '{'; i = 0; out = []
    while True:
        j = s.find(tag, i)
        if j < 0: return out
        k = j + len(tag); d = 1
        while k < len(s) and d:
            if s[k] == '{': d += 1
            elif s[k] == '}': d -= 1
            if d: k += 1
        out.append((j, j + len(tag), k)); i = k + 1

def delete_cmd(s, cmd):
    out = []; last = 0
    for a, b, c in spans(s, cmd):
        out.append(s[last:a]); last = c + 1
    out.append(s[last:]); return ''.join(out)

def unwrap_cmd(s, cmd):
    out = []; last = 0
    for a, b, c in spans(s, cmd):
        out.append(s[last:a]); out.append(s[b:c]); last = c + 1
    out.append(s[last:]); return ''.join(out)

def norm(s):
    return ' '.join(re.sub(r'%.*', '', s).split()).split()

prod = open(PROD).read()
for c in ('Gala', 'R'):
    while '\\' + c + '{' in prod: prod = delete_cmd(prod, c)
while '\\B{' in prod: prod = unwrap_cmd(prod, 'B')   # leaked markers: unwrap, don't delete

draft_raw = open(DRAFT).read()
draft_flat = draft_raw
while '\\B{' in draft_flat: draft_flat = unwrap_cmd(draft_flat, 'B')

A, B = norm(prod), norm(draft_flat)
eq = [False] * len(B)
for t, x1, x2, y1, y2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
    if t == 'equal':
        for y in range(y1, y2): eq[y] = True

# map each \B block (in the ORIGINAL draft) onto a word range of the flattened draft
blocks = spans(draft_raw, 'B')
pre_words = {}
for a, b, c in blocks:
    head = draft_raw[:a]
    while '\\B{' in head: head = unwrap_cmd(head, 'B')
    start = len(norm(head))
    body = draft_raw[b:c]
    while '\\B{' in body: body = unwrap_cmd(body, 'B')
    w = norm(body)
    line = draft_raw[:a].count('\n') + 1
    seg = eq[start:start + len(w)]
    frac = sum(seg) / len(seg) if seg else 1.0
    pre_words[(a, b, c)] = (line, len(w), frac, ' '.join(w)[:110])

def in_comment(pos):
    """True if this \\B{ sits inside a % comment (documentation, not draft prose)."""
    bol = draft_raw.rfind('\n', 0, pos) + 1
    seg = draft_raw[bol:pos]
    return '%' in re.sub(r'\\%', '', seg)

print(f"{'line':>5} {'words':>5} {'=prod':>6}  verdict   text")
stale = live = partial = skipped = 0
for (a, b, c), (line, n, frac, txt) in pre_words.items():
    if in_comment(a): skipped += 1; continue
    if n == 0: v = 'EMPTY'
    elif frac > 0.995:
        # ⚠ short blocks align spuriously (paper_writer.md sec.2, trap 3).
        v = 'STALE?' if n < 8 else 'STALE'; stale += 1
    elif frac < 0.05: v = 'LIVE '; live += 1
    else: v = 'PARTIAL'; partial += 1
    print(f"{line:>5} {n:>5} {frac:>6.2f}  {v:<8}  {txt}")
print(f"\nSTALE(unwrap) {stale}   LIVE(keep) {live}   PARTIAL(inspect) {partial}"
      f"   [{skipped} in comments, skipped]")
if stale:
    print("\n⚠ 'STALE?' = block under 8 words. Do NOT unwrap on this verdict alone:\n"
          "  grep production.tex for a distinctive token from the block and require a\n"
          "  NON-ZERO count first. A lone \\cite{} aligns spuriously (trap 3).")
