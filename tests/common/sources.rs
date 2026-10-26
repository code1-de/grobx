//! The workspace's sources as test code reads them (rule-breaches.15).
//! Each scan reads the code of the `.rs` files under directories of
//! `grobx/`, with comments and the insides of literals blanked, by a rule
//! of `CLAUDE.md` as `iterations/0089-rule-breaches/plan.md`, _The rules as
//! the item reads them_, reads it. A scan reads tokens and takes no types:
//! what it cannot see, the item's inventory holds.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The formatting macros, each with the position of its format string
/// among its arguments (rule-breaches.1).
const FORMATTING: [(&str, usize); 18] = [
    ("format", 0),
    ("format_args", 0),
    ("print", 0),
    ("println", 0),
    ("eprint", 0),
    ("eprintln", 0),
    ("panic", 0),
    ("unreachable", 0),
    ("todo", 0),
    ("unimplemented", 0),
    ("write", 1),
    ("writeln", 1),
    ("assert", 1),
    ("debug_assert", 1),
    ("assert_eq", 2),
    ("assert_ne", 2),
    ("debug_assert_eq", 2),
    ("debug_assert_ne", 2),
];

/// The assertion macros, inside which a `matches!` stays
/// (rule-breaches.5).
const ASSERTIONS: [&str; 6] = [
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
];

/// The macros that panic where they run, which production code calls only
/// where rule-breaches.8 keeps them.
const PANICS: [&str; 6] = [
    "assert",
    "assert_eq",
    "assert_ne",
    "panic",
    "todo",
    "unimplemented",
];

/// The first parts of the paths of the std group of `use`s
/// (rule-breaches.2).
const STD: [&str; 3] = ["std", "core", "alloc"];

/// The first parts of the paths of the local group of `use`s, beside the
/// crate's own modules.
const LOCAL: [&str; 3] = ["crate", "super", "self"];

/// The enums of other crates that the workspace chooses between, each with
/// its variants: std's `Ordering` and num-bigint's `Sign`.
const FOREIGN: [(&str, &[&str]); 2] = [
    ("Ordering", &["Less", "Equal", "Greater"]),
    ("Sign", &["Minus", "NoSign", "Plus"]),
];

/// The variants of `Option` and `Result`, which a `match` names as
/// patterns of their own (rule-breaches.3).
const OPTION_RESULT: [&str; 4] = ["Some", "None", "Ok", "Err"];

/// The directory `grobx/`.
fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// A `.rs` file of `grobx/`.
pub struct Source {
    /// The path from `grobx/`, its parts joined by `/`.
    pub path: String,
    /// The text as it stands.
    pub text: String,
}

impl Source {
    /// A source of `text` at `path`, for a scan of written-out text.
    pub fn new(path: &str, text: &str) -> Source {
        Source {
            path: path.to_string(),
            text: text.to_string(),
        }
    }

    /// Whether the file is test code: named `*_tests.rs`, or under a
    /// directory `tests`.
    pub fn is_test(&self) -> bool {
        self.path.ends_with("_tests.rs") || self.path.split('/').any(|part| part == "tests")
    }

    /// The crate the file belongs to: its package's directory, or for a
    /// file of a test binary the path up to its directory `tests`.
    pub fn krate(&self) -> String {
        let parts: Vec<&str> = self.path.split('/').collect();
        match parts.iter().position(|&part| part == "tests") {
            Some(k) => parts[..=k].join("/"),
            None => self.package().to_string(),
        }
    }

    /// The directory of the package the file belongs to: a member's, or
    /// `""` for the root package's.
    pub fn package(&self) -> &str {
        let first = self.path.split('/').next().unwrap_or("");
        if members().iter().any(|m| m == first) {
            first
        } else {
            ""
        }
    }
}

/// Every `.rs` file under the directories `dirs` of `grobx/`, by path. A
/// directory `target` is left out.
pub fn rust_sources(dirs: &[&str]) -> Vec<Source> {
    let mut found = Vec::new();
    let mut pending: Vec<PathBuf> = dirs.iter().map(|d| root().join(d)).collect();
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            let entries = fs::read_dir(&path)
                .unwrap_or_else(|e| panic!("{path}: {e}", path = path.display()));
            for entry in entries {
                pending.push(entry.expect("a directory entry").path());
            }
            continue;
        }
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{path}: {e}", path = path.display()));
        let relative = path.strip_prefix(root()).expect("a path under grobx/");
        let parts: Vec<String> = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        found.push(Source {
            path: parts.join("/"),
            text,
        });
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found
}

/// The directories of every package's code: the root package's `src` and
/// `tests`, and each member's.
pub fn workspace() -> Vec<String> {
    let mut dirs = vec!["src".to_string(), "tests".to_string()];
    dirs.extend(members().iter().cloned());
    dirs
}

/// Every `.rs` file of the workspace (`workspace`).
pub fn workspace_sources() -> Vec<Source> {
    let dirs = workspace();
    let dirs: Vec<&str> = dirs.iter().map(String::as_str).collect();
    rust_sources(&dirs)
}

/// The text of the manifest of the package in the directory `package` of
/// `grobx/`, `""` for the root package's.
pub fn manifest(package: &str) -> String {
    let path = root().join(package).join("Cargo.toml");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}", path = path.display()))
}

/// The keys of a manifest's table `[table]`: per line up to the next table,
/// the text before its first `=` or `.`.
pub fn table_keys(manifest: &str, table: &str) -> Vec<String> {
    let header = format!("[{table}]");
    manifest
        .lines()
        .skip_while(|l| l.trim() != header)
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split(['=', '.']).next().unwrap_or("").trim().to_string())
        .collect()
}

/// The quoted value of `key` in a manifest's table `[table]`.
fn table_value(manifest: &str, table: &str, key: &str) -> Option<String> {
    let header = format!("[{table}]");
    manifest
        .lines()
        .skip_while(|l| l.trim() != header)
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
        })
}

/// The directories of the workspace's members, as the root manifest's
/// `members` names them.
fn members() -> &'static [String] {
    static MEMBERS: OnceLock<Vec<String>> = OnceLock::new();
    MEMBERS.get_or_init(|| {
        let text = manifest("");
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with("members"))
            .expect("the root manifest names its members");
        line.split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    })
}

/// The crates a package's code names as external in its `use`s: its
/// dependencies of every kind, and its own package and library for its
/// tests, each with `_` for `-`.
pub fn external_crates(package: &str) -> BTreeSet<String> {
    let text = manifest(package);
    let mut names: BTreeSet<String> = ["dependencies", "dev-dependencies", "build-dependencies"]
        .iter()
        .flat_map(|table| table_keys(&text, table))
        .collect();
    for table in ["package", "lib"] {
        names.extend(table_value(&text, table, "name"));
    }
    names.into_iter().map(|n| n.replace('-', "_")).collect()
}

// --- the lexer ---

/// A string literal of a source.
pub struct Literal {
    /// The byte of its first character, its prefix or its opening quote.
    pub start: usize,
    /// Its text between its quotes, as written.
    pub text: String,
}

/// A source's text twice, each of the text's length with its line breaks
/// kept: `code` with every comment and the inside of every literal blanked,
/// `uncommented` with the comments alone blanked; and its string literals.
pub struct Lexed {
    pub code: String,
    pub uncommented: String,
    pub literals: Vec<Literal>,
}

/// Whether `c` may stand in an identifier.
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The bytes of a UTF-8 character that `lead` opens.
fn char_len(lead: u8) -> usize {
    match lead.leading_ones() {
        2 => 2,
        3 => 3,
        4 => 4,
        _ => 1,
    }
}

/// Blanks `bytes[from..to]` but its line breaks.
fn blank(bytes: &mut [u8], from: usize, to: usize) {
    let to = to.min(bytes.len());
    for b in &mut bytes[from..to] {
        if *b != b'\n' {
            *b = b' ';
        }
    }
}

/// Where a raw string opens at `at`: the byte past its opening quote and
/// the count of its `#`s, for `r"`, `r#"`, `br"`, and `cr"`.
fn raw_open(b: &[u8], at: usize) -> Option<(usize, usize)> {
    let mut i = at;
    if b.get(i).is_some_and(|&c| c == b'b' || c == b'c') {
        i += 1;
    }
    if b.get(i) != Some(&b'r') {
        return None;
    }
    i += 1;
    let hashes = b[i..].iter().take_while(|&&c| c == b'#').count();
    i += hashes;
    (b.get(i) == Some(&b'"')).then_some((i + 1, hashes))
}

/// The byte past a character literal whose quote stands at `q`, or None
/// where the quote opens a lifetime or a label.
fn char_end(b: &[u8], q: usize) -> Option<usize> {
    let next = *b.get(q + 1)?;
    if next == b'\\' {
        let close = b[q + 3..].iter().position(|&c| c == b'\'')?;
        return Some(q + 3 + close + 1);
    }
    let len = char_len(next);
    (b.get(q + 1 + len) == Some(&b'\'')).then_some(q + 2 + len)
}

/// `text` read into its code, its uncommented text, and its literals.
pub fn lex(text: &str) -> Lexed {
    let b = text.as_bytes();
    let mut code = b.to_vec();
    let mut uncommented = b.to_vec();
    let mut literals = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let joined = i > 0 && is_ident(b[i - 1]);
        if b[i..].starts_with(b"//") {
            let end = b[i..]
                .iter()
                .position(|&c| c == b'\n')
                .map_or(b.len(), |k| i + k);
            blank(&mut code, i, end);
            blank(&mut uncommented, i, end);
            i = end;
            continue;
        }
        if b[i..].starts_with(b"/*") {
            let (mut depth, mut j) = (0usize, i);
            while j < b.len() {
                if b[j..].starts_with(b"/*") {
                    depth += 1;
                    j += 2;
                } else if b[j..].starts_with(b"*/") {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            blank(&mut code, i, j);
            blank(&mut uncommented, i, j);
            i = j;
            continue;
        }
        if !joined {
            if let Some((open, hashes)) = raw_open(b, i) {
                let mut close = open;
                while close < b.len()
                    && !(b[close] == b'"' && b[close + 1..].iter().take(hashes).all(|&c| c == b'#'))
                {
                    close += 1;
                }
                literals.push(Literal {
                    start: i,
                    text: text[open..close].to_string(),
                });
                blank(&mut code, open, close);
                i = close + 1 + hashes;
                continue;
            }
            let quoted = b[i] == b'"';
            let prefixed = (b[i] == b'b' || b[i] == b'c') && b.get(i + 1) == Some(&b'"');
            if quoted || prefixed {
                let open = if quoted { i + 1 } else { i + 2 };
                let mut close = open;
                while close < b.len() && b[close] != b'"' {
                    close += if b[close] == b'\\' { 2 } else { 1 };
                }
                literals.push(Literal {
                    start: i,
                    text: text[open..close.min(b.len())].to_string(),
                });
                blank(&mut code, open, close);
                i = close + 1;
                continue;
            }
            let quote = if b[i] == b'b' && b.get(i + 1) == Some(&b'\'') {
                Some(i + 1)
            } else {
                (b[i] == b'\'').then_some(i)
            };
            if let Some((q, past)) = quote.and_then(|q| char_end(b, q).map(|past| (q, past))) {
                blank(&mut code, q + 1, past - 1);
                i = past;
                continue;
            }
        }
        i += 1;
    }
    Lexed {
        code: String::from_utf8(code).expect("blanks of whole characters keep UTF-8"),
        uncommented: String::from_utf8(uncommented).expect("blanks of whole characters keep UTF-8"),
        literals,
    }
}

/// The line, from 1, of the byte at `at`.
pub fn line_of(text: &str, at: usize) -> usize {
    text[..at.min(text.len())].matches('\n').count() + 1
}

/// Whether `code` holds `prefix` at `at`.
fn starts(code: &[u8], at: usize, prefix: &[u8]) -> bool {
    code.get(at..).is_some_and(|rest| rest.starts_with(prefix))
}

/// The first byte at or past `at` that is no white space.
fn skip_space(code: &[u8], at: usize) -> usize {
    at + code[at.min(code.len())..]
        .iter()
        .take_while(|c| c.is_ascii_whitespace())
        .count()
}

/// The byte past the last byte before `end` that is no white space.
fn trim_back(code: &[u8], end: usize) -> usize {
    end - code[..end]
        .iter()
        .rev()
        .take_while(|c| c.is_ascii_whitespace())
        .count()
}

/// The bracket that closes the one at `open`, or `code.len()`. The three
/// kinds count alike: the code's literals and comments are blanked.
fn close(code: &[u8], open: usize) -> usize {
    let mut depth = 0usize;
    for (k, &c) in code.iter().enumerate().skip(open) {
        if b"([{".contains(&c) {
            depth += 1;
        } else if b")]}".contains(&c) {
            depth -= 1;
            if depth == 0 {
                return k;
            }
        }
    }
    code.len()
}

/// The bracket that opens the one at `closing`, or 0.
fn open_before(code: &[u8], closing: usize) -> usize {
    let mut depth = 0usize;
    for k in (0..=closing).rev() {
        if b")]}".contains(&code[k]) {
            depth += 1;
        } else if b"([{".contains(&code[k]) {
            depth -= 1;
            if depth == 0 {
                return k;
            }
        }
    }
    0
}

/// Every place in `code` where `word` stands with no identifier joined to
/// either side.
fn words(code: &[u8], word: &str) -> Vec<usize> {
    let w = word.as_bytes();
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(k) = code[from..].windows(w.len()).position(|x| x == w) {
        let at = from + k;
        let before = at > 0 && is_ident(code[at - 1]);
        let after = code.get(at + w.len()).is_some_and(|&c| is_ident(c));
        if !before && !after {
            found.push(at);
        }
        from = at + 1;
    }
    found
}

/// Whether `word` stands at `at` with no identifier joined after it.
fn word_at(code: &[u8], at: usize, word: &str) -> bool {
    code[at.min(code.len())..].starts_with(word.as_bytes())
        && !code.get(at + word.len()).is_some_and(|&c| is_ident(c))
}

/// The identifier that starts at `at`, or "".
fn ident_at(code: &[u8], at: usize) -> &str {
    let len = code[at.min(code.len())..]
        .iter()
        .take_while(|&&c| is_ident(c))
        .count();
    std::str::from_utf8(&code[at..at + len]).expect("an identifier is ASCII")
}

/// The path that starts at `at`: its parts, `::` between them.
fn path_at(code: &[u8], at: usize) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut i = at;
    loop {
        let part = ident_at(code, i);
        if part.is_empty() {
            break;
        }
        parts.push(part);
        i += part.len();
        if !starts(code, i, b"::") {
            break;
        }
        i += 2;
    }
    parts
}

/// The path that ends at `end`: its parts, `::` between them.
fn path_before(code: &[u8], end: usize) -> Vec<&str> {
    let mut start = end;
    while start > 0 && (is_ident(code[start - 1]) || code[..start].ends_with(b"::")) {
        start -= if is_ident(code[start - 1]) { 1 } else { 2 };
    }
    path_at(code, start)
}

/// The ranges of the arguments between the brackets at `open` and
/// `closing`, split at the commas outside nested brackets.
fn arguments(code: &[u8], open: usize, closing: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = open + 1;
    let mut k = open + 1;
    while k < closing {
        if b"([{".contains(&code[k]) {
            k = close(code, k) + 1;
            continue;
        }
        if code[k] == b',' {
            out.push((from, k));
            from = k + 1;
        }
        k += 1;
    }
    if code[from..closing].iter().any(|c| !c.is_ascii_whitespace()) {
        out.push((from, closing));
    }
    out
}

/// A macro call: its name, where its name stands, and its brackets.
struct Call<'a> {
    name: &'a str,
    at: usize,
    open: usize,
    close: usize,
}

/// Every call of a macro named in `names`.
fn calls<'a>(code: &[u8], names: &[&'a str]) -> Vec<Call<'a>> {
    let mut out = Vec::new();
    for &name in names {
        for at in words(code, name) {
            let bang = at + name.len();
            if code.get(bang) != Some(&b'!') {
                continue;
            }
            let open = skip_space(code, bang + 1);
            if code.get(open).is_some_and(|c| b"([{".contains(c)) {
                out.push(Call {
                    name,
                    at,
                    open,
                    close: close(code, open),
                });
            }
        }
    }
    out.sort_by_key(|c| c.at);
    out
}

/// The ranges of the `#[cfg(test)]` items of `code`, each from its
/// attribute to its end: an item with a body to the body's close and a
/// comma after it, any other to its `;` or `,`, or before the close of the
/// block it stands in.
fn test_items(code: &[u8]) -> Vec<(usize, usize)> {
    let attribute = b"#[cfg(test)]";
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(k) = code[from..]
        .windows(attribute.len())
        .position(|w| w == attribute)
    {
        let start = from + k;
        let mut i = skip_space(code, start + attribute.len());
        while starts(code, i, b"#[") {
            i = skip_space(code, close(code, i + 1) + 1);
        }
        let mut end = code.len();
        while i < code.len() {
            match code[i] {
                b'(' | b'[' => i = close(code, i) + 1,
                b'{' => {
                    let after = skip_space(code, close(code, i) + 1);
                    end = if code.get(after) == Some(&b',') {
                        after + 1
                    } else {
                        close(code, i) + 1
                    };
                    break;
                }
                b';' | b',' => {
                    end = i + 1;
                    break;
                }
                b'}' | b')' | b']' => {
                    end = i;
                    break;
                }
                _ => i += 1,
            }
        }
        out.push((start, end));
        from = end.max(start + 1);
    }
    out
}

/// The ranges of the `const` items of `code`, which compute at compile
/// time: each from `const` to its `;`.
fn const_items(code: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for at in words(code, "const") {
        let line = code[..at]
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(0, |k| k + 1);
        let lead = std::str::from_utf8(&code[line..at])
            .expect("code is UTF-8")
            .trim();
        if !lead.is_empty() && !lead.starts_with("pub") {
            continue;
        }
        let name = skip_space(code, at + "const".len());
        if word_at(code, name, "fn") {
            continue;
        }
        let mut i = name;
        while i < code.len() && code[i] != b';' {
            i = if b"([{".contains(&code[i]) {
                close(code, i) + 1
            } else {
                i + 1
            };
        }
        out.push((at, i));
    }
    out
}

/// A source's lexed text with its `#[cfg(test)]` items blanked as well:
/// the code that production builds compile.
pub fn production(source: &Source) -> Lexed {
    let mut lexed = lex(&source.text);
    let mut code = std::mem::take(&mut lexed.code).into_bytes();
    let mut uncommented = std::mem::take(&mut lexed.uncommented).into_bytes();
    for (from, to) in test_items(&code) {
        blank(&mut code, from, to);
        blank(&mut uncommented, from, to);
    }
    lexed.code = String::from_utf8(code).expect("blanks keep UTF-8");
    lexed.uncommented = String::from_utf8(uncommented).expect("blanks keep UTF-8");
    lexed
}

/// Each source's finds as `<path>:<line>`, or `<path>:<line> <what>`.
fn places(
    sources: &[Source],
    mut scan: impl FnMut(&Source) -> Vec<(usize, String)>,
) -> Vec<String> {
    let mut out = Vec::new();
    for source in sources {
        for (line, what) in scan(source) {
            let path = &source.path;
            out.push(if what.is_empty() {
                format!("{path}:{line}")
            } else {
                format!("{path}:{line} {what}")
            });
        }
    }
    out
}

// --- format strings (rule-breaches.1) ---

/// Whether the format string `text` holds a positional placeholder: `{}`,
/// `{:…}`, or `{0}`.
pub fn positional(text: &str) -> bool {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let doubled = b.get(i + 1) == Some(&b[i]);
        match b[i] {
            b'{' | b'}' if doubled => i += 2,
            b'{' => {
                let end = text[i..].find('}').map_or(b.len(), |k| i + k);
                let name = text[i + 1..end].split(':').next().unwrap_or("");
                if name.bytes().all(|c| c.is_ascii_digit()) {
                    return true;
                }
                i = end + 1;
            }
            _ => i += 1,
        }
    }
    false
}

/// The lines of `source`'s format strings that hold a positional
/// placeholder.
pub fn positional_formats_in(source: &Source) -> Vec<(usize, String)> {
    let lexed = lex(&source.text);
    let code = lexed.code.as_bytes();
    let names: Vec<&str> = FORMATTING.iter().map(|&(name, _)| name).collect();
    let mut lines = BTreeSet::new();
    for call in calls(code, &names) {
        let position = FORMATTING
            .iter()
            .find(|&&(name, _)| name == call.name)
            .map_or(0, |&(_, p)| p);
        let Some(&(from, _)) = arguments(code, call.open, call.close).get(position) else {
            continue;
        };
        let start = skip_space(code, from);
        let format = lexed.literals.iter().find(|l| l.start == start);
        if format.is_some_and(|l| positional(&l.text)) {
            lines.insert(line_of(&source.text, start));
        }
    }
    lines
        .into_iter()
        .map(|line| (line, String::new()))
        .collect()
}

/// Every format string of `sources` with a positional placeholder, as
/// `<path>:<line>`.
pub fn positional_formats(sources: &[Source]) -> Vec<String> {
    places(sources, positional_formats_in)
}

// --- `use` declarations (rule-breaches.2) ---

/// An item of a module's body, as the order of its `use`s reads it.
enum Entry {
    /// A `use`: its first line, attributes included, its last line, its
    /// group, and the first part of its path.
    Use {
        first: usize,
        last: usize,
        group: usize,
        head: String,
    },
    /// A `mod` declaration without a body.
    Declaration,
    /// Any other item, a `macro_rules!` macro with its name.
    Item(Option<String>),
}

/// The group of a `use` whose path opens with `head`: 0 std, 1 external, 2
/// local.
fn group(head: &str, external: &BTreeSet<String>) -> usize {
    if STD.contains(&head) {
        0
    } else if LOCAL.contains(&head) || !external.contains(head) {
        2
    } else {
        1
    }
}

/// The byte past the visibility at `at`: `pub`, `pub(crate)`, and the like.
fn past_visibility(code: &[u8], at: usize) -> usize {
    if !word_at(code, at, "pub") {
        return at;
    }
    let after = skip_space(code, at + "pub".len());
    if code.get(after) == Some(&b'(') {
        skip_space(code, close(code, after) + 1)
    } else {
        after
    }
}

/// The byte past the item that starts at `at`, which ends at its first `;`
/// or at the close of its first body.
fn item_end(code: &[u8], at: usize, to: usize) -> usize {
    let mut i = at;
    while i < to {
        match code[i] {
            b'(' | b'[' => i = close(code, i) + 1,
            b'{' => return close(code, i) + 1,
            b';' => return i + 1,
            _ => i += 1,
        }
    }
    to
}

/// Each `use` that opens a statement in `code[from..to]`, the body of a
/// function or a block: one inside a body.
fn body_uses(code: &[u8], text: &str, from: usize, to: usize, out: &mut Vec<(usize, String)>) {
    for at in words(&code[..to], "use") {
        if at < from {
            continue;
        }
        let before = trim_back(code, at);
        if before > 0 && b";{}]".contains(&code[before - 1]) {
            out.push((line_of(text, at), "a use inside a body".to_string()));
        }
    }
}

/// The places of `code[from..to]`, a module's body, whose `use`s break the
/// order of rule-breaches.2; and those of the modules inside it.
fn module_uses(
    code: &[u8],
    text: &str,
    from: usize,
    to: usize,
    external: &BTreeSet<String>,
    out: &mut Vec<(usize, String)>,
) {
    let mut entries = Vec::new();
    let mut i = from;
    loop {
        i = skip_space(code, i);
        if i >= to {
            break;
        }
        if code[i] == b';' {
            i += 1;
            continue;
        }
        let start = i;
        while starts(code, i, b"#[") || starts(code, i, b"#![") {
            let open = if code[i + 1] == b'!' { i + 2 } else { i + 1 };
            i = skip_space(code, close(code, open) + 1);
        }
        if i >= to {
            break;
        }
        let at = past_visibility(code, i);
        if word_at(code, at, "use") {
            let end = item_end(code, at, to);
            let path = skip_space(code, at + "use".len());
            let path = if starts(code, path, b"::") {
                path + 2
            } else {
                path
            };
            let head = ident_at(code, path).to_string();
            entries.push(Entry::Use {
                first: line_of(text, start),
                last: line_of(text, end.saturating_sub(1)),
                group: group(&head, external),
                head,
            });
            i = end;
            continue;
        }
        if word_at(code, at, "mod") {
            let mut k = at;
            while k < to && code[k] != b';' && code[k] != b'{' {
                k += 1;
            }
            if code.get(k) == Some(&b'{') {
                let end = close(code, k);
                module_uses(code, text, k + 1, end, external, out);
                entries.push(Entry::Item(None));
                i = end + 1;
            } else {
                entries.push(Entry::Declaration);
                i = k + 1;
            }
            continue;
        }
        let end = item_end(code, at, to);
        let name = word_at(code, at, "macro_rules")
            .then(|| ident_at(code, skip_space(code, at + "macro_rules!".len())).to_string());
        body_uses(code, text, at, end, out);
        entries.push(Entry::Item(name));
        i = end;
    }
    let mut item = false;
    let mut macros = BTreeSet::new();
    let mut previous: Option<(usize, usize)> = None;
    for entry in entries {
        match entry {
            Entry::Use {
                first,
                last,
                group,
                head,
            } => {
                if item && !macros.contains(&head) {
                    out.push((first, "a use after an item".to_string()));
                    previous = None;
                    continue;
                }
                if let Some((before, before_group)) = previous {
                    let blank = text
                        .lines()
                        .skip(before)
                        .take(first.saturating_sub(before + 1))
                        .any(|l| l.trim().is_empty());
                    if group < before_group {
                        out.push((first, format!("group {group} after group {before_group}")));
                    } else if group > before_group && !blank {
                        out.push((first, format!("no blank line before group {group}")));
                    } else if group == before_group && blank {
                        out.push((first, format!("a blank line inside group {group}")));
                    }
                }
                previous = Some((last, group));
            }
            Entry::Declaration => previous = None,
            Entry::Item(name) => {
                item = true;
                previous = None;
                macros.extend(name);
            }
        }
    }
}

/// The places of `source`'s `use`s that break the order of
/// rule-breaches.2, each with what breaks it; `external` names the
/// package's external crates.
pub fn misplaced_uses_in(source: &Source, external: &BTreeSet<String>) -> Vec<(usize, String)> {
    let lexed = lex(&source.text);
    let code = lexed.code.as_bytes();
    let mut out = Vec::new();
    module_uses(code, &source.text, 0, code.len(), external, &mut out);
    out.sort();
    out
}

/// Every `use` of `sources` out of the order of rule-breaches.2, as
/// `<path>:<line> <what>`.
pub fn misplaced_uses(sources: &[Source]) -> Vec<String> {
    let mut external: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    places(sources, |source| {
        let package = source.package().to_string();
        let crates = external
            .entry(package.clone())
            .or_insert_with(|| external_crates(&package));
        misplaced_uses_in(source, crates)
    })
}

// --- choices between variants (rule-breaches.3 to .5) ---

// rule-breaches.3 design -- a choice on Ordering takes its methods is_lt, is_le, is_gt, is_ge, is_eq, and is_ne, and every other choice between an enum's variants is a match that names every variant, not a match with three arms for each comparison of an Ordering (not measured)
// rule-breaches.4 design -- a comparison of an Option with Some of a number or of a reference to data compares values and stays, and one with Some(true), Some(false), or Some of a variant chooses a variant and becomes a match, not every comparison with Some a match (not measured)
// rule-breaches.5 design -- an assertion that expects one variant stays, not the assertions written as matches (not measured)

/// An enum the sources define.
struct Enum {
    variants: BTreeSet<String>,
    non_exhaustive: bool,
    krate: String,
}

/// The enums of the sources by name, with std's and num-bigint's that the
/// workspace chooses between.
pub struct Enums(BTreeMap<String, Vec<Enum>>);

impl Enums {
    /// The enums `sources` define, and those of `FOREIGN`.
    pub fn of(sources: &[Source]) -> Enums {
        let mut enums: BTreeMap<String, Vec<Enum>> = BTreeMap::new();
        for (name, variants) in FOREIGN {
            enums.entry(name.to_string()).or_default().push(Enum {
                variants: variants.iter().map(ToString::to_string).collect(),
                non_exhaustive: false,
                krate: "another crate".to_string(),
            });
        }
        for source in sources {
            let lexed = lex(&source.text);
            let code = lexed.code.as_bytes();
            for at in words(code, "enum") {
                let name = ident_at(code, skip_space(code, at + "enum".len()));
                let Some(k) = code[at..].iter().position(|&c| c == b'{' || c == b';') else {
                    continue;
                };
                let open = at + k;
                if name.is_empty() || code[open] != b'{' {
                    continue;
                }
                let variants = arguments(code, open, close(code, open))
                    .into_iter()
                    .filter_map(|(from, _)| {
                        let mut i = skip_space(code, from);
                        while starts(code, i, b"#[") {
                            i = skip_space(code, close(code, i + 1) + 1);
                        }
                        let variant = ident_at(code, i);
                        (!variant.is_empty()).then(|| variant.to_string())
                    })
                    .collect();
                let item = code[..at]
                    .iter()
                    .rposition(|&c| b";{}".contains(&c))
                    .map_or(0, |k| k + 1);
                let head = std::str::from_utf8(&code[item..at]).expect("code is UTF-8");
                enums.entry(name.to_string()).or_default().push(Enum {
                    variants,
                    non_exhaustive: head.contains("non_exhaustive"),
                    krate: source.krate(),
                });
            }
        }
        Enums(enums)
    }

    /// The enum whose variant the path's last two parts name.
    fn variant<'a>(&self, path: &[&'a str]) -> Option<&'a str> {
        let [.., name, variant] = path else {
            return None;
        };
        let known = self
            .0
            .get(*name)
            .is_some_and(|defs| defs.iter().any(|d| d.variants.contains(*variant)));
        known.then_some(*name)
    }

    /// Whether `named` holds every variant of an enum named `name`, or of
    /// `Option` or `Result`.
    fn covered(&self, name: &str, named: &BTreeSet<String>) -> bool {
        let all = |variants: &[&str]| variants.iter().all(|v| named.contains(*v));
        match name {
            "Option" => all(&["Some", "None"]),
            "Result" => all(&["Ok", "Err"]),
            _ => self.0.get(name).is_some_and(|defs| {
                defs.iter()
                    .any(|d| d.variants.iter().all(|v| named.contains(v)))
            }),
        }
    }

    /// Whether every enum named `name` is another crate's
    /// `#[non_exhaustive]` enum, seen from the crate `from`, whose `_` arm
    /// `CLAUDE.md` allows.
    fn open_to(&self, name: &str, from: &str) -> bool {
        self.0
            .get(name)
            .is_some_and(|defs| defs.iter().all(|d| d.non_exhaustive && d.krate != from))
    }
}

/// Whether the code at `at` opens a variant that `==` or `!=` chooses:
/// `None`, `Some(true)`, `Some(false)`, `Some(<variant>)`, or `<variant>`.
fn chooses_at(code: &[u8], at: usize, enums: &Enums) -> bool {
    if word_at(code, at, "None") {
        return true;
    }
    if starts(code, at, b"Some(") {
        let mut inner = skip_space(code, at + "Some(".len());
        if code.get(inner) == Some(&b'&') {
            inner += 1;
        }
        return word_at(code, inner, "true")
            || word_at(code, inner, "false")
            || enums.variant(&path_at(code, inner)).is_some();
    }
    enums.variant(&path_at(code, at)).is_some()
}

/// Whether the code that ends at `end` closes such a variant.
fn chooses_before(code: &[u8], end: usize, enums: &Enums) -> bool {
    if end == 0 {
        return false;
    }
    if code[end - 1] == b')' {
        let open = open_before(code, end - 1);
        return path_before(code, open) == ["Some"]
            && chooses_at(code, path_start(code, open), enums);
    }
    let path = path_before(code, end);
    path == ["None"] || enums.variant(&path).is_some()
}

/// The first byte of the path that ends at `end`.
fn path_start(code: &[u8], end: usize) -> usize {
    let mut start = end;
    while start > 0 && (is_ident(code[start - 1]) || code[..start].ends_with(b"::")) {
        start -= if is_ident(code[start - 1]) { 1 } else { 2 };
    }
    start
}

/// The ranges of the calls of assertion macros in `code`.
fn assertion_ranges(code: &[u8]) -> Vec<(usize, usize)> {
    calls(code, &ASSERTIONS)
        .into_iter()
        .map(|call| (call.open, call.close))
        .collect()
}

/// The pattern of the `let` at `at`: from past `let` to its `=`.
fn let_pattern(code: &[u8], at: usize) -> (usize, usize) {
    let from = at + "let".len();
    let mut k = from;
    while k < code.len() {
        match code[k] {
            b'(' | b'[' | b'{' => k = close(code, k) + 1,
            b'=' if !b"=>".contains(&code[k + 1]) && !b"=!<>".contains(&code[k - 1]) => break,
            b';' => break,
            _ => k += 1,
        }
    }
    (from, k)
}

/// Whether `code[from..to]` names a variant of an enum that `CLAUDE.md`
/// asks a `match` to name in full, seen from the crate `krate`.
fn names_variant(code: &[u8], from: usize, to: usize, enums: &Enums, krate: &str) -> bool {
    let mut i = from;
    while i < to {
        if is_ident(code[i]) && (i == 0 || !is_ident(code[i - 1])) {
            let path = path_at(code, i);
            if let Some(name) = enums.variant(&path)
                && !enums.open_to(name, krate)
            {
                return true;
            }
            i += path.iter().map(|p| p.len() + 2).sum::<usize>();
            continue;
        }
        i += 1;
    }
    false
}

/// The places of `source` where a choice between an enum's variants is
/// made by `==`, `!=`, `matches!` outside an assertion, or a `let` that
/// chooses: `if let`, `while let`, a `let` after `&&`, and `let … else`.
pub fn variant_choices_in(source: &Source, enums: &Enums) -> Vec<(usize, String)> {
    let lexed = lex(&source.text);
    let code = lexed.code.as_bytes();
    let text = &source.text;
    let mut out = Vec::new();
    for k in 1..code.len().saturating_sub(2) {
        let operator = (code[k] == b'=' || code[k] == b'!') && code[k + 1] == b'=';
        if !operator || b"=!<>".contains(&code[k - 1]) || code[k + 2] == b'=' {
            continue;
        }
        let right = skip_space(code, k + 2);
        if chooses_at(code, right, enums) || chooses_before(code, trim_back(code, k), enums) {
            out.push((line_of(text, k), "a comparison with a variant".to_string()));
        }
    }
    let assertions = assertion_ranges(code);
    for call in calls(code, &["matches"]) {
        if assertions
            .iter()
            .any(|&(from, to)| from < call.at && call.at < to)
        {
            continue;
        }
        let Some(&(from, to)) = arguments(code, call.open, call.close).get(1) else {
            continue;
        };
        let (from, to) = bare_pattern(code, from, to);
        let named = pattern_enum(code, from, to, enums);
        if named.is_some_and(|name| !enums.open_to(&name, &source.krate())) {
            out.push((
                line_of(text, call.at),
                "matches! outside an assertion".to_string(),
            ));
        }
    }
    for at in words(code, "let") {
        let (from, to) = let_pattern(code, at);
        if !names_variant(code, from, to, enums, &source.krate()) {
            continue;
        }
        let before = trim_back(code, at);
        let conditional = path_before(code, before)
            .last()
            .is_some_and(|w| *w == "if" || *w == "while")
            || code[..before].ends_with(b"&&");
        let end = item_end(code, to, code.len());
        let otherwise = !words(&code[to..end], "else").is_empty();
        if conditional || otherwise {
            out.push((
                line_of(text, at),
                "a let that chooses a variant".to_string(),
            ));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every choice of `sources` between an enum's variants by `==`, `!=`,
/// `matches!`, or a `let`, as `<path>:<line> <what>`.
pub fn variant_choices(sources: &[Source], enums: &Enums) -> Vec<String> {
    places(sources, |source| variant_choices_in(source, enums))
}

/// The patterns of the arms of the `match` whose block stands between
/// `open` and `closing`, each with the byte it starts at.
fn arms(code: &[u8], open: usize, closing: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = open + 1;
    loop {
        i = skip_space(code, i);
        while i < closing && starts(code, i, b"#[") {
            i = skip_space(code, close(code, i + 1) + 1);
        }
        if i >= closing {
            break;
        }
        let start = i;
        while i < closing && !starts(code, i, b"=>") {
            i = if b"([{".contains(&code[i]) {
                close(code, i) + 1
            } else {
                i + 1
            };
        }
        out.push((start, i));
        i = skip_space(code, i + 2);
        if code.get(i) == Some(&b'{') {
            i = skip_space(code, close(code, i) + 1);
            if code.get(i) == Some(&b',') {
                i += 1;
            }
            continue;
        }
        while i < closing && code[i] != b',' {
            i = if b"([{".contains(&code[i]) {
                close(code, i) + 1
            } else {
                i + 1
            };
        }
        i += 1;
    }
    out
}

/// `code[from..to]` split at the bytes `sep` outside nested brackets, each
/// part trimmed.
fn split_outside(code: &[u8], from: usize, to: usize, sep: u8) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut start, mut k) = (from, from);
    while k < to {
        if b"([{".contains(&code[k]) {
            k = close(code, k) + 1;
            continue;
        }
        if code[k] == sep {
            out.push((start, k));
            start = k + 1;
        }
        k += 1;
    }
    out.push((start, to));
    out.into_iter()
        .map(|(a, b)| {
            (
                skip_space(code, a),
                trim_back(code, b).max(skip_space(code, a)),
            )
        })
        .collect()
}

/// The pattern `code[from..to]` without its guard and its binding `x @`.
fn bare_pattern(code: &[u8], from: usize, to: usize) -> (usize, usize) {
    let (start, end, _) = guarded_pattern(code, from, to);
    (start, end)
}

/// The pattern `code[from..to]` without its guard and its binding `x @`,
/// and whether it has a guard.
fn guarded_pattern(code: &[u8], from: usize, to: usize) -> (usize, usize, bool) {
    let mut end = to;
    let mut guarded = false;
    let mut k = from;
    while k < to {
        if b"([{".contains(&code[k]) {
            k = close(code, k) + 1;
            continue;
        }
        if word_at(code, k, "if") && (k == 0 || !is_ident(code[k - 1])) {
            end = trim_back(code, k);
            guarded = true;
            break;
        }
        k += 1;
    }
    let mut start = skip_space(code, from);
    if code[start] == b'|' {
        start = skip_space(code, start + 1);
    }
    if let Some(at) = code[start..end].iter().position(|&c| c == b'@') {
        start = skip_space(code, start + at + 1);
    }
    (start, end, guarded)
}

/// The variants that the unguarded arms among `patterns` name at their
/// top, each the last part of its path.
fn named_variants(code: &[u8], patterns: &[(usize, usize, bool)]) -> BTreeSet<String> {
    let mut named = BTreeSet::new();
    for &(a, b, guarded) in patterns {
        if guarded {
            continue;
        }
        for (c, _) in split_outside(code, a, b, b'|') {
            if let Some(last) = path_at(code, c).last() {
                named.insert((*last).to_string());
            }
        }
    }
    named
}

/// The enum a pattern names a variant of: `Option` or `Result` for their
/// variants, the enum's name for another, None for none.
fn pattern_enum(code: &[u8], from: usize, to: usize, enums: &Enums) -> Option<String> {
    for (a, b) in split_outside(code, from, to, b'|') {
        let mut a = a;
        while code.get(a).is_some_and(|&c| c == b'&')
            || word_at(code, a, "ref")
            || word_at(code, a, "mut")
        {
            a = skip_space(code, a + if code[a] == b'&' { 1 } else { 3 });
        }
        if a >= b {
            continue;
        }
        if code[a] == b'(' {
            let inner = close(code, a);
            for (c, d) in split_outside(code, a + 1, inner, b',') {
                if let Some(name) = pattern_enum(code, c, d, enums) {
                    return Some(name);
                }
            }
            continue;
        }
        let head = ident_at(code, a);
        if OPTION_RESULT.contains(&head) {
            return Some(
                if head == "Ok" || head == "Err" {
                    "Result"
                } else {
                    "Option"
                }
                .to_string(),
            );
        }
        if let Some(name) = enums.variant(&path_at(code, a)) {
            return Some(name.to_string());
        }
    }
    None
}

/// Whether the pattern `code[from..to]` takes every value: `_`, or a bare
/// binding.
fn catch_all(code: &[u8], from: usize, to: usize) -> bool {
    let pattern = std::str::from_utf8(&code[from..to])
        .expect("code is UTF-8")
        .trim();
    pattern == "_"
        || (pattern.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
            && pattern.bytes().all(is_ident)
            && pattern != "true"
            && pattern != "false")
}

/// The places of `source` where a `match` on an enum takes a `_` arm or a
/// bare binding for the variants it does not name, or a `_` in a tuple
/// where another arm names a variant; another package's
/// `#[non_exhaustive]` enum left out.
pub fn wildcard_arms_in(source: &Source, enums: &Enums) -> Vec<(usize, String)> {
    let lexed = lex(&source.text);
    let code = lexed.code.as_bytes();
    let text = &source.text;
    let mut out = Vec::new();
    for at in words(code, "match") {
        let mut open = at + "match".len();
        while open < code.len() && code[open] != b'{' {
            open = if code[open] == b'(' || code[open] == b'[' {
                close(code, open) + 1
            } else if code[open] == b';' {
                code.len()
            } else {
                open + 1
            };
        }
        if open >= code.len() {
            continue;
        }
        let patterns: Vec<(usize, usize, bool)> = arms(code, open, close(code, open))
            .into_iter()
            .map(|(a, b)| guarded_pattern(code, a, b))
            .collect();
        let named = patterns
            .iter()
            .find_map(|&(a, b, _)| pattern_enum(code, a, b, enums));
        let Some(name) = named else {
            continue;
        };
        if enums.open_to(&name, &source.krate()) {
            continue;
        }
        let covered = enums.covered(&name, &named_variants(code, &patterns));
        for &(a, b, guarded) in &patterns {
            if catch_all(code, a, b) && !(guarded && covered) {
                out.push((line_of(text, a), format!("a catch-all arm on {name}")));
            }
        }
        let tuples: Vec<Vec<(usize, usize)>> = patterns
            .iter()
            .filter(|&&(a, _, _)| code.get(a) == Some(&b'('))
            .map(|&(a, _, _)| split_outside(code, a + 1, close(code, a), b','))
            .collect();
        for tuple in &tuples {
            for (k, &(c, d)) in tuple.iter().enumerate() {
                let opposite = tuples.iter().any(|other| {
                    other
                        .get(k)
                        .is_some_and(|&(e, f)| pattern_enum(code, e, f, enums).is_some())
                });
                if code[c..d] == *b"_" && opposite {
                    out.push((line_of(text, c), format!("a _ in a tuple on {name}")));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every `_` arm and catch-all binding of `sources` on an enum, as
/// `<path>:<line> <what>`.
pub fn wildcard_arms(sources: &[Source], enums: &Enums) -> Vec<String> {
    places(sources, |source| wildcard_arms_in(source, enums))
}

// --- panics (rule-breaches.6 to .8) ---

/// The macros of `PANICS` that `source`'s production code calls, outside
/// its `#[cfg(test)]` items and its `const` items; none in test code.
pub fn production_asserts_in(source: &Source) -> Vec<(usize, String)> {
    if source.is_test() {
        return Vec::new();
    }
    let lexed = production(source);
    let mut code = lexed.code.into_bytes();
    for (from, to) in const_items(&code) {
        blank(&mut code, from, to);
    }
    calls(&code, &PANICS)
        .into_iter()
        .map(|call| {
            (
                line_of(&source.text, call.at),
                format!("{name}!", name = call.name),
            )
        })
        .collect()
}

/// Every call of a macro of `PANICS` in `sources`' production code, as
/// `<path>:<line> <macro>`.
pub fn production_asserts(sources: &[Source]) -> Vec<String> {
    places(sources, production_asserts_in)
}

// --- suppressions (rule-breaches.13, .14) ---

/// A lint suppression.
pub struct Allow {
    /// Where it stands, as `<path>:<line>`.
    pub place: String,
    /// Whether it is `#![allow(…)]`, the module's.
    pub inner: bool,
    /// The lints it names.
    pub lints: Vec<String>,
    /// Whether it gives a reason.
    pub reason: bool,
}

/// Every `#[allow(…)]` and `#![allow(…)]` of `source`.
pub fn allows_in(source: &Source) -> Vec<Allow> {
    let lexed = lex(&source.text);
    let code = lexed.code.as_bytes();
    let mut out = Vec::new();
    for at in words(code, "allow") {
        let inner = code[..at].ends_with(b"#![");
        if !inner && !code[..at].ends_with(b"#[") {
            continue;
        }
        let open = at + "allow".len();
        if code.get(open) != Some(&b'(') {
            continue;
        }
        let parts = arguments(code, open, close(code, open));
        let texts: Vec<String> = parts
            .iter()
            .map(|&(a, b)| {
                std::str::from_utf8(&code[a..b])
                    .expect("code is UTF-8")
                    .trim()
                    .to_string()
            })
            .collect();
        let line = line_of(&source.text, at);
        let path = &source.path;
        out.push(Allow {
            place: format!("{path}:{line}"),
            inner,
            lints: texts
                .iter()
                .filter(|w| !w.starts_with("reason"))
                .cloned()
                .collect(),
            reason: texts.iter().any(|w| w.starts_with("reason")),
        });
    }
    out
}

/// Every lint suppression of `sources`.
pub fn allows(sources: &[Source]) -> Vec<Allow> {
    sources.iter().flat_map(allows_in).collect()
}

// --- literals and definitions (rule-breaches.10, .11) ---

/// The lines of the file `path` of `grobx/` whose code, comments left out,
/// holds one of `needles`, each as `<path>:<line> <needle>`.
pub fn literals(path: &str, needles: &[&str]) -> Vec<String> {
    let text = fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
    let lexed = lex(&text);
    let mut out = Vec::new();
    for (k, line) in lexed.uncommented.lines().enumerate() {
        for needle in needles {
            if line.contains(needle) {
                let number = k + 1;
                out.push(format!("{path}:{number} {needle}"));
            }
        }
    }
    out
}

/// The lines of `sources`' production code, comments and `#[cfg(test)]`
/// items left out, of which `holds` holds, as `<path>:<line>`.
pub fn lines_where(sources: &[Source], holds: impl Fn(&str) -> bool) -> Vec<String> {
    places(sources, |source| {
        if source.is_test() {
            return Vec::new();
        }
        production(source)
            .uncommented
            .lines()
            .enumerate()
            .filter(|&(_, line)| holds(line))
            .map(|(k, _)| (k + 1, String::new()))
            .collect()
    })
}
