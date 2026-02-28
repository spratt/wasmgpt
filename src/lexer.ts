// WAT lexer: tokenizes WAT text into a flat list of token strings.
// Comments are stripped. Tokens are returned exactly as they appear
// in the source text (case-sensitive, no normalization).

export function tokenize(text: string): Array<string> {
  const tokens = new Array<string>();
  let pos: i32 = 0;
  const len: i32 = text.length;

  while (pos < len) {
    pos = skipWhitespace(text, pos, len);
    if (pos >= len) break;

    const ch = text.charCodeAt(pos);

    // Line comment: ;;
    if (ch == 59 && pos + 1 < len && text.charCodeAt(pos + 1) == 59) {
      pos = skipLineComment(text, pos, len);
      continue;
    }

    // Block comment: (; ... ;)
    if (ch == 40 && pos + 1 < len && text.charCodeAt(pos + 1) == 59) {
      pos = skipBlockComment(text, pos, len);
      continue;
    }

    // Open paren
    if (ch == 40) {
      tokens.push("(");
      pos++;
      continue;
    }

    // Close paren
    if (ch == 41) {
      tokens.push(")");
      pos++;
      continue;
    }

    // String literal
    if (ch == 34) {
      const end = readStringLiteral(text, pos, len);
      tokens.push(text.substring(pos, end));
      pos = end;
      continue;
    }

    // Atom: identifier, mnemonic, number, keyword, type
    const end = readAtom(text, pos, len);
    tokens.push(text.substring(pos, end));
    pos = end;
  }

  return tokens;
}

function isWhitespace(ch: i32): bool {
  return ch == 32 || ch == 9 || ch == 10 || ch == 13;
}

function isTerminator(ch: i32): bool {
  return isWhitespace(ch) || ch == 40 || ch == 41 || ch == 34 || ch == 59;
}

function skipWhitespace(text: string, pos: i32, len: i32): i32 {
  let i = pos;
  while (i < len && isWhitespace(text.charCodeAt(i))) {
    i++;
  }
  return i;
}

function skipLineComment(text: string, pos: i32, len: i32): i32 {
  let i = pos + 2;
  while (i < len && text.charCodeAt(i) != 10) {
    i++;
  }
  return i;
}

function skipBlockComment(text: string, pos: i32, len: i32): i32 {
  let i = pos + 2;
  let depth: i32 = 1;
  while (i + 1 < len && depth > 0) {
    const c0 = text.charCodeAt(i);
    const c1 = text.charCodeAt(i + 1);
    if (c0 == 40 && c1 == 59) {
      depth++;
      i += 2;
    } else if (c0 == 59 && c1 == 41) {
      depth--;
      i += 2;
    } else {
      i++;
    }
  }
  return i;
}

function readStringLiteral(text: string, pos: i32, len: i32): i32 {
  let i = pos + 1;
  while (i < len) {
    const ch = text.charCodeAt(i);
    if (ch == 92) {
      i += 2;
      continue;
    }
    if (ch == 34) {
      return i + 1;
    }
    i++;
  }
  return i;
}

function readAtom(text: string, pos: i32, len: i32): i32 {
  let i = pos;
  while (i < len && !isTerminator(text.charCodeAt(i))) {
    i++;
  }
  return i;
}
