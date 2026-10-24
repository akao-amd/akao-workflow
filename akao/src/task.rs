//! `akao task <nick> <name> <desc>`: make a worker's next task dir, `<root>/<NNNN>_<desc>`,
//! optionally with the controller's brief in it as BRIEF.md, and print its path.
//!
//! The root is the container's own (`$AKAO_ARTIFACT_ROOT`, else the working directory every
//! `docker exec` starts in, if under /<year>/), read inside the container, so it is right
//! whatever the week.  The last line of output is `task dir <path>`.

use crate::exec::{q, Runner};
use crate::state;
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

pub struct Options {
    pub nick: String,
    pub name: String,
    pub desc: String,
    /// The task number as the user gave it (`7070`), instead of the next one.
    pub number: Option<String>,
    /// A local file to put into the task dir as BRIEF.md.
    pub brief: Option<String>,
}

/// bash, in the container: `$1` the number or empty, `$2` the description, `$3` 1 to read
/// BRIEF.md from stdin, `$4` 1 to only print the dir it would make.
const SCRIPT: &str = r#"set -e
root=$AKAO_ARTIFACT_ROOT
if [ -z "$root" ]; then
    root=$PWD
    case $root in
    /[0-9][0-9][0-9][0-9]/?*) ;;
    *) echo "no artifact root: AKAO_ARTIFACT_ROOT is unset and the working directory $root is not under /<year>/" >&2; exit 1 ;;
    esac
fi
case $root in
/*) ;;
*) echo "artifact root $root is not absolute" >&2; exit 1 ;;
esac
[ -d "$root" ] || { echo "artifact root $root is not a directory" >&2; exit 1; }
n=$1
if [ -z "$n" ]; then
    last=0
    for d in "$root"/[0-9][0-9][0-9][0-9]_*/; do
        [ -d "$d" ] || continue
        b=${d%/}; b=${b##*/}; v=$((10#${b:0:4}))
        if [ "$v" -gt "$last" ]; then last=$v; fi
    done
    [ "$last" -lt 9999 ] || { echo "$root has task 9999; give --number" >&2; exit 1; }
    n=$(printf %04d $((last + 1)))
fi
dir=$root/${n}_$2
for d in "$root/${n}"_*; do
    [ -d "$d" ] || continue
    echo "task number $n is taken: $d" >&2
    exit 1
done
if [ "$4" = 1 ]; then echo "$dir"; exit 0; fi
mkdir "$dir"
if [ "$3" = 1 ]; then
    cat >"$dir/.BRIEF.md.part" || { echo "$dir made, but its BRIEF.md did not arrive" >&2; exit 1; }
    mv "$dir/.BRIEF.md.part" "$dir/BRIEF.md" || { echo "$dir made, but its BRIEF.md could not be installed" >&2; exit 1; }
fi
"#;

pub fn run(state: &state::State, r: &Runner, opts: &Options) -> Result<()> {
    let host = state.host(&opts.nick)?;
    let name = opts.name.strip_prefix("akao_").unwrap_or(&opts.name);
    if !state::valid_name(name) {
        bail!("invalid container name '{}'", opts.name);
    }
    if !state::valid_name(&opts.desc) {
        bail!(
            "the task's description must match [A-Za-z0-9][A-Za-z0-9._-]*, got '{}'",
            opts.desc
        );
    }
    let mut number = opts.number.clone().unwrap_or_default();
    if opts.number.is_some() && !(number.len() == 4 && number.bytes().all(|b| b.is_ascii_digit()) && number != "0000") {
        bail!("--number must be four digits, 0001-9999 (e.g. 7070), got '{number}'");
    }
    let brief = if let Some(b) = &opts.brief {
        if !Path::new(b).is_file() {
            bail!("brief {b} is not a file");
        }
        Some(fs::read(b).with_context(|| format!("cannot read brief {b}"))?)
    } else {
        None
    };
    let container = format!("akao_{name}");
    let endpoint = format!("ssh://{}", host.nick);
    let exec = |number: &str, interactive: bool, read_brief: bool, dry: bool| -> Vec<String> {
        let mut argv: Vec<String> = ["docker", "-H", &endpoint, "exec"].map(String::from).into();
        if interactive {
            argv.push("-i".into());
        }
        argv.extend([
            container.clone(),
            "bash".into(),
            "-c".into(),
            SCRIPT.into(),
            "task".into(),
        ]);
        argv.extend([
            number.to_string(),
            opts.desc.clone(),
            if read_brief { "1" } else { "0" }.into(),
            if dry { "1" } else { "0" }.into(),
        ]);
        argv
    };
    // Its path first, read-only: the dry run's answer, and a taken number fails here.  Then
    // made with that number, checked again next to the mkdir: best effort against a second
    // controller making a task in the same root meanwhile (one controller drives a worker).
    let dir = r
        .query(&exec(&number, false, false, true))
        .with_context(|| format!("cannot read {container}'s artifact root on {}", host.nick))?;
    let dir = dir.trim();
    number = dir.rsplit('/').next().unwrap_or_default().chars().take(4).collect();
    if r.dry_run {
        println!(
            "would make {dir}{} (a preview: the number may be taken by then)",
            if opts.brief.is_some() { " with BRIEF.md" } else { "" }
        );
        return Ok(());
    }
    match &brief {
        Some(bytes) => r.input(&exec(&number, true, true, false), bytes)?,
        None => r.run(&exec(&number, false, false, false))?,
    }
    println!("next: akao open {} {name} --cwd {}", host.nick, q(dir));
    println!("task dir {dir}");
    Ok(())
}
