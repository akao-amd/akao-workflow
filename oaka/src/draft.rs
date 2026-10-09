//! `oaka draft`: write a plan.toml to start from.
//!
//! The draft is a guess: whatever the caller (often an agent with more context) passed as
//! hints, plus what oaka sees on the machine.  Everything left open is a comment listing
//! the choices, and `oaka check` names each field that still needs a value.

use crate::plan::{valid_server_name, PLAN};
use crate::profile::Library;
use crate::sys;
use anyhow::{bail, Result};
use std::fs;
use std::path::Path;

pub struct Options {
    pub profiles: Vec<String>,
    pub gpus: Vec<u32>,
    pub clients: Vec<String>,
    pub force: bool,
}

pub const CLIENT_KINDS: &[&str] = &["gsm8k", "fixed-seq"];

fn server_name(profile: &str, taken: &[String]) -> String {
    let recipe = profile.rsplit('/').next().unwrap_or("main");
    let mut base: String = recipe
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    if !valid_server_name(&base) {
        base = format!("s{base}");
    }
    let mut name = base.clone();
    let mut n = 2;
    while taken.contains(&name) {
        name = format!("{base}{n}");
        n += 1;
    }
    name
}

fn toml_list<T: ToString>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(T::to_string).collect::<Vec<_>>().join(", "))
}

fn client_block(kind: &str, server: &str, commented: bool) -> String {
    let body = match kind {
        "gsm8k" => format!(
            "[[client]]\nkind = \"gsm8k\"            # accuracy gate: the run stops below min_score\n\
             server = \"{server}\"\nthinking = false\nmin_score = 0.90\n"
        ),
        _ => format!(
            "[[client]]\nkind = \"fixed-seq\"        # InferenceX fixed-seq-len throughput\n\
             server = \"{server}\"\nisl_osl = [[1024, 1024], [8192, 1024]]\nconc = [4, 8, 16, 32, 64]\n\
             range_ratio = 0.8\nrepeats = 1\n"
        ),
    };
    if commented {
        body.lines().map(|l| format!("# {l}\n")).collect()
    } else {
        body
    }
}

pub fn render(lib: &Library, opts: &Options) -> Result<String> {
    for k in &opts.clients {
        if !CLIENT_KINDS.contains(&k.as_str()) {
            bail!("unknown client kind {k:?}; one of {}", CLIENT_KINDS.join(" "));
        }
    }
    let gpus = sys::gpus();
    let arch = sys::arch();
    let all = lib.list()?;
    let mut fitting = Vec::new();
    for name in &all {
        let p = lib.resolve(name)?;
        if arch.as_ref().is_none_or(|a| p.arch.is_empty() || p.arch.contains(a)) {
            fitting.push(name.clone());
        }
    }

    let mut out = String::new();
    out += "# oaka plan.  Edit, then: oaka check / oaka compile / oaka run\n";
    match &gpus {
        Some(g) if !g.is_empty() => {
            out += &format!(
                "# this machine: {} GPUs (0-{}), {}\n",
                g.len(),
                g.len() - 1,
                arch.as_deref().unwrap_or("mixed archs")
            )
        }
        _ => out += "# this machine: no ROCm GPUs found\n",
    }
    out += &format!("# library: {}\n", lib.root.display());
    out += &format!(
        "# profiles{}: {}\n",
        arch.as_ref().map(|a| format!(" for {a}")).unwrap_or_default(),
        if fitting.is_empty() {
            "<none>".to_string()
        } else {
            fitting.join(" ")
        }
    );
    let models = sys::models();
    if !models.is_empty() {
        out += &format!("# models under /model: {}\n", models.join(" "));
    }

    let profiles = if opts.profiles.is_empty() {
        vec![String::new()]
    } else {
        opts.profiles.clone()
    };
    let mut names: Vec<String> = Vec::new();
    let mut free = opts.gpus.clone();
    for profile in &profiles {
        if !profile.is_empty() {
            lib.resolve(profile)?;
        }
        let name = if profile.is_empty() {
            "main".to_string()
        } else {
            server_name(profile, &names)
        };
        // GPUs: one server takes all it was given; several take tp (pinned) or 1 each, in order.
        let want = if profiles.len() == 1 {
            free.len()
        } else if profile.is_empty() {
            1
        } else {
            lib.resolve(profile)?.tp.unwrap_or(1) as usize
        };
        let mine: Vec<u32> = if free.len() >= want {
            free.drain(..want).collect()
        } else {
            Vec::new()
        };
        out += "\n[[server]]\n";
        out += &format!("name = \"{name}\"\n");
        out += &format!(
            "profile = \"{profile}\"{}\n",
            if profile.is_empty() {
                "             # REQUIRED: one of the profiles above"
            } else {
                ""
            }
        );
        out += &format!(
            "gpus = {}{}\n",
            toml_list(&mine),
            if mine.is_empty() {
                "                   # REQUIRED: GPU indices; --tp = their count"
            } else {
                ""
            }
        );
        out += "# port = 29911            # default: a free port in 29900-30050, kept in plan.lock.toml\n";
        out += "# model = \"/model/...\"    # default: the profile's model\n";
        out += "[server.env]               # overrides on top of the profile; NAME = false unsets\n";
        out += "[server.args]              # launch_server flags without --; true = bare flag, false = drop\n";
        names.push(name);
    }

    out += "\n";
    if opts.clients.is_empty() {
        out += "# clients run in this order once every server is ready; uncomment what you need.\n";
        for kind in CLIENT_KINDS {
            out += "\n";
            out += &client_block(kind, &names[0], true);
        }
    } else {
        for kind in &opts.clients {
            for name in &names {
                out += "\n";
                out += &client_block(kind, name, false);
            }
        }
    }
    Ok(out)
}

pub fn run(dir: &Path, lib: &Library, opts: &Options) -> Result<()> {
    let path = dir.join(PLAN);
    if path.exists() && !opts.force {
        bail!("{} exists; use --force to overwrite it", path.display());
    }
    let text = render(lib, opts)?;
    fs::write(&path, &text)?;
    print!("{text}");
    eprintln!("oaka: wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan;

    #[test]
    fn drafts_parse_and_hinted_ones_check() {
        let root = std::env::temp_dir().join(format!("oaka-draft-{}-{}", std::process::id(), fastrand::u32(..)));
        let lib = Library { root: root.clone() };
        fs::create_dir_all(lib.profiles_dir().join("m")).unwrap();
        fs::write(lib.profile_path("m/triton"), "model = '/model/m'\n").unwrap();
        fs::write(lib.profile_path("m/tp2"), "model = '/model/m'\ntp = 2\n").unwrap();

        let bare = render(
            &lib,
            &Options {
                profiles: vec![],
                gpus: vec![],
                clients: vec![],
                force: false,
            },
        )
        .unwrap();
        let p: plan::Plan = toml::from_str(&bare).unwrap();
        assert!(p.clients.is_empty());
        let e = plan::check_plan(&root, p, &lib, None).err().unwrap();
        assert!(format!("{e:#}").contains("profile is empty"));

        let opts = Options {
            profiles: vec!["m/triton".into(), "m/tp2".into()],
            gpus: vec![5, 6, 7],
            clients: vec!["gsm8k".into(), "fixed-seq".into()],
            force: false,
        };
        let text = render(&lib, &opts).unwrap();
        let checked = plan::check_plan(&root, toml::from_str(&text).unwrap(), &lib, None).unwrap();
        assert_eq!(checked.servers[0].gpus, [5]);
        assert_eq!(checked.servers[1].gpus, [6, 7]);
        assert_eq!(checked.servers[1].name, "tp2");
        assert_eq!(checked.plan.clients.len(), 4);
        fs::remove_dir_all(root).unwrap();
    }
}
