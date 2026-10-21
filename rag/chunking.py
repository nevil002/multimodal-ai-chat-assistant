import re
from nltk.tokenize import sent_tokenize
from config import MAX_CHUNK_SIZE, OVERLAP
from rag.schemas import Result


HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s*([A-Z][^\n]{2,100})$")
BULLETS = ("#", "•", "●", "–", "- ", "* ")


def split_large_text(text,max_size):
    """
    Split large text into chunks of a maximum size
    """
    if len(text) <= max_size:
        return [text]

    sentences=sent_tokenize(text)
    pieces=[]
    current_piece=""
    for sentence in sentences:
        if len(current_piece) + len(sentence) +1<= max_size:
            if current_piece:
                current_piece+=" "
            current_piece+=sentence
        else:
            if current_piece:
                pieces.append(current_piece)
            current_piece=sentence  
    if current_piece:
        pieces.append(current_piece)
    return pieces

def has_md_headings(text):
    return bool(re.search(r"(?m)^#{1,6}\s", text))

def normalize_text(text):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.fullmatch(r"\d{1,3}", s):                    # bare page numbers
            continue
        if re.search(r"\.{5,}\s*\d*\s*$", s):              # TOC dot leaders
            continue
        m = HEADING_RE.match(s)
        if m and not s.endswith(".") and len(s) < 100:     # numbered headings -> markdown
            out.append("#" * min(m.group(1).count(".") + 1, 6) + " " + s)
        else:
            out.append(s)
    return "\n".join(out)

def looks_hard_wrapped(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = [p for p in paras if not p.startswith(BULLETS + ("|",))]
    if len(paras) < 20:
        return False
    unfinished = sum(1 for p in paras
                     if "\n" not in p and not p.endswith((".", "!", "?", ":")))
    return unfinished / len(paras) > 0.4

def reflow(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if p.startswith("|"):                              # tables stand alone
            if buf: out.append(buf); buf = ""
            out.append(p)
            continue
        if buf and (p.startswith(BULLETS) or re.match(r"^\d+[.)]\s", p)
                    or buf.endswith((".", "!", "?", ":", ";"))):
            out.append(buf); buf = ""
        buf = f"{buf} {p}" if buf else p
        if p.startswith("#"):                              # headings stand alone
            out.append(buf); buf = ""
    if buf:
        out.append(buf)
    return "\n\n".join(out)

def prepare_text(text):
    if not has_md_headings(text):
        text = normalize_text(text)
        if looks_hard_wrapped(text):
            text = reflow(text)
    return text

import re  # add to your imports

def is_table_line(line):
    return line.lstrip().startswith("|")

def split_blocks(body):
    """Split section text into ('table'|'text', block) pairs, in order."""
    blocks, buf, kind = [], [], None

    def flush():
        nonlocal buf, kind
        text = "\n".join(buf).strip()
        if text:
            blocks.append((kind, text))
        buf, kind = [], None

    for line in body.split("\n"):
        if not line.strip():          # blank line ends the current block
            flush() 
            continue
        line_kind = "table" if is_table_line(line) else "text"
        if kind and line_kind != kind:
            flush()
        kind = line_kind
        buf.append(line)
    flush()
    return blocks

def split_table(table, max_size):
    """Split a big markdown table by rows, repeating the header in each piece."""
    if len(table) <= max_size:
        return [table]
    lines = table.split("\n")
    has_sep = len(lines) > 1 and "-" in lines[1] and not re.search(r"[A-Za-z0-9]", lines[1])
    header = "\n".join(lines[:2] if has_sep else lines[:1])
    rows = lines[2:] if has_sep else lines[1:]

    pieces, current = [], header
    for row in rows:
        if current != header and len(current) + len(row) + 1 > max_size:
            pieces.append(current)
            current = header
        current += "\n" + row
    if current != header:
        pieces.append(current)
    return pieces

def chunk_section(body):
    """Pack blocks into chunks. Returns [{'text': str, 'table': bool}]."""
    chunks, current, has_table = [], "", False

    def flush():
        nonlocal current, has_table
        if current:
            chunks.append({"text": current, "table": has_table})
        current, has_table = "", False

    for kind, block in split_blocks(body):
        if len(block) > MAX_CHUNK_SIZE:       # oversized block: split on its own
            flush()
            pieces = (split_table(block, MAX_CHUNK_SIZE) if kind == "table"
                      else split_large_text(block, MAX_CHUNK_SIZE))
            chunks.extend({"text": p, "table": kind == "table"} for p in pieces)    
            continue
        if current and len(current) + len(block) + 2 > MAX_CHUNK_SIZE:
            flush()
        current = f"{current}\n\n{block}" if current else block
        has_table = has_table or kind == "table"
    flush()
    return chunks

def overlap_tail(prev, overlap=OVERLAP):
    units = [u.strip() for u in re.split(r"\n\s*\n", prev) if u.strip()]
    tail, size = [], 0
    for unit in reversed(units):
        if size + len(unit) > overlap:
            break
        tail.insert(0, unit)
        size += len(unit)
    if not tail and units:                    # last paragraph too long: use its last sentences
        for s in reversed(sent_tokenize(units[-1])):
            if size + len(s) > overlap:
                break
            tail.insert(0, s)
            size += len(s)
    return "\n\n".join(tail)

def add_overlap(chunks, overlap=OVERLAP):
    out = []
    for i, c in enumerate(chunks):
        text = c["text"]
        if i > 0 and not c["table"] and not chunks[i - 1]["table"]:
            tail = overlap_tail(chunks[i - 1]["text"], overlap)
            if tail:
                text = f"{tail}\n\n{text}"
        out.append(text)
    return out

def split_markdown_text(text):
    sections = re.split(r"(?m)(?=^#{1,6}\s)", text)
    stack, result = {}, []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", section)
        if m:
            level = len(m.group(1))
            stack = {l: t for l, t in stack.items() if l < level}
            stack[level] = m.group(2).strip()
            body = section[m.end():].strip()
        else:
            body = section
        path = " > ".join(stack[l] for l in sorted(stack))
        for chunk in add_overlap(chunk_section(body)):
            result.append(f"[{path}]\n\n{chunk}" if path else chunk)
    return result


def process_document(document):
    """
    Process a document and create chunks
    """
    print(f"\n processing:{document["source"]}")        
    print(f"Total characters in the particular document={len(document["text"])}")
    raw_chunks=split_markdown_text(document["text"])
    chunks=[]
    for raw_chunk in raw_chunks:
        chunks.append(Result(page_content=raw_chunk, metadata={"source":document["source"], "type":document["type"]}))
    return chunks