//! `akao init <nick> <name>`: bring up worker container akao_<name> on a remote box.
//!
//! Every step checks what already exists and reuses it, so re-running init on a
//! half-initialized worker resumes instead of failing.

use crate::exec::{q, show, Runner, ESCALATE};
use crate::state::{self, Host, State};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// The fixed part of every `docker run`; per-host mounts and rest args follow it.
const DOCKER_RUN_SKELETON: &[&str] = &[
    "run",
    "--rm",
    "-d",
    "--privileged",
    "--ulimit",
    "nofile=1048576",
    "--network=host",
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add",
    "video",
    "--cap-add=SYS_PTRACE",
    "--security-opt",
    "seccomp=unconfined",
    "-e",
    "PYTHONPATH=",
    "-e",
    "LANG=C.UTF-8",
    "-e",
    "LC_ALL=C.UTF-8",
    "-e",
    "TERM=tmux-256color",
    "--ipc=host",
    "--shm-size=32g",
];

const APT_PACKAGES: &[&str] = &["vim", "less", "tmux", "docker.io", "git"];

/// Excluded from the control-plane deploy.
const DEPLOY_EXCLUDES: &[&str] = &[".git", "__pycache__", "*.pyc", ".claude"];

pub struct Options {
    pub nick: String,
    pub name: String,
    pub week: Option<String>,
    /// The worker's artifact root, instead of /<year>/<week>/<name>.
    pub artifact_root: Option<String>,
    /// Image instead of the host's or the default (`akao mirror`: the entry's).
    pub image: Option<String>,
    pub skip_setup: bool,
}

/// Everything init needs, resolved from the arguments and state up front.
pub struct Plan {
    pub host: Host,
    /// docker context name (= nick)
    pub context: String,
    /// akao_<name>
    pub container: String,
    pub image: String,
    pub year: String,
    /// The worker's artifact root, /<year>/<week>/<name> unless given: its working directory
    /// and $AKAO_ARTIFACT_ROOT in the container; the same path under <host_home> on the host.
    pub artifact_root: String,
    /// <host_home>/container_home/akao_<name>
    pub host_container_home: String,
}

impl Plan {
    pub fn new(state: &State, opts: &Options) -> Result<Plan> {
        if !state::valid_name(&opts.name) {
            bail!("invalid container name '{}'", opts.name);
        }
        let name = opts.name.strip_prefix("akao_").unwrap_or(&opts.name);
        let host = state.host(&opts.nick)?;
        let image = match (&opts.image, &host.image) {
            (Some(i), _) | (None, Some(i)) => i.clone(),
            (None, None) => state.require("default_image")?,
        };
        let year = state::work_year();
        let artifact_root = match (&opts.artifact_root, &opts.week) {
            (Some(_), Some(_)) => bail!("give --week or --artifact-root, not both"),
            (Some(root), None) => {
                check_artifact_root(root, &year)?;
                root.trim_end_matches('/').to_string()
            }
            (None, week) => {
                let week = week.clone().unwrap_or_else(state::work_week);
                if !(week.len() == 4 && week.starts_with("ww") && week[2..].bytes().all(|b| b.is_ascii_digit())) {
                    bail!("week must look like ww41, got '{week}'");
                }
                format!("/{year}/{week}/{name}")
            }
        };
        let container = format!("akao_{name}");
        Ok(Plan {
            context: host.nick.clone(),
            artifact_root,
            host_container_home: format!("{}/container_home/{container}", host.home()),
            container,
            image,
            year,
            host,
        })
    }

    fn host_year_dir(&self) -> String {
        format!("{}/{}", self.host.home(), self.year)
    }

    pub fn docker(&self, args: &[&str]) -> Vec<String> {
        let mut argv: Vec<String> = vec!["docker".into(), "--context".into(), self.context.clone()];
        argv.extend(args.iter().map(|s| s.to_string()));
        argv
    }

    pub fn docker_run_argv(&self) -> Result<Vec<String>> {
        let mut argv = self.docker(DOCKER_RUN_SKELETON);
        let h = &self.host;
        argv.push(format!("--name={}", self.container));
        for (src, dst) in [
            (h.model_path.clone(), "/model".to_string()),
            (h.docker_sock().to_string(), "/var/run/docker.sock".to_string()),
            (self.host_year_dir(), format!("/{}", self.year)),
            (self.host_container_home.clone(), "/root".to_string()),
        ] {
            argv.extend(["-v".into(), format!("{src}:{dst}")]);
        }
        argv.extend([
            "-e".into(),
            format!("{}={}", state::ARTIFACT_ROOT_ENV, self.artifact_root),
            "-e".into(),
            format!("{}={}", state::REPO_ROOT_ENV, state::WORKER_REPO),
        ]);
        argv.extend(["-w".into(), self.artifact_root.clone()]);
        argv.extend(h.rest_args()?);
        argv.extend([self.image.clone(), "sleep".into(), "infinity".into()]);
        Ok(argv)
    }

    fn exec(&self, opts: &[&str], script: &str) -> Vec<String> {
        let mut args = vec!["exec"];
        args.extend(opts);
        args.extend([self.container.as_str(), "bash", "-c", script]);
        self.docker(&args)
    }
}

/// `tar` of `paths` in `dir` to stdout, owned root.  Deploys dereference symlinks
/// (`deref`): a link into the console's tree would dangle on a box.
fn tar_create(dir: &Path, paths: &[String], excludes: &[&str], deref: bool) -> Vec<String> {
    let mut argv: Vec<String> = vec!["tar".into(), "-C".into(), dir.display().to_string()];
    if deref {
        argv.push("-h".into());
    }
    argv.extend(["--owner=0".into(), "--group=0".into()]);
    argv.extend(excludes.iter().map(|e| format!("--exclude={e}")));
    argv.push("-czf".into());
    argv.push("-".into());
    argv.extend(paths.iter().cloned());
    argv
}

/// Where the repo bundle lands in the container on its way into the clone.
const REPO_BUNDLE: &str = "/root/.cache/akao-workflow.bundle";

/// Bash, run in the worker: clone the bundle into `repo` the first time; later fetch it into
/// `refs/remotes/console/*` and fast-forward `main` only while the clone is on main, clean
/// and behind.  Never fatal for the worker's own state; says what it did.
pub fn repo_sync_script(bundle: &str, repo: &str, origin: Option<&str>) -> String {
    let (b, r) = (q(bundle), q(repo));
    let set_origin = origin
        .map(|o| format!("git -C \"$R\" remote set-url origin {}; ", q(o)))
        .unwrap_or_default();
    format!(
        r#"B={b}; R={r}; say() {{ echo "  $*"; }}
fetch() {{ git -C "$R" fetch -q "$B" '+refs/heads/*:refs/remotes/console/*'; }}
at() {{ git -C "$R" log -1 --format='%h %s' "$1"; }}
if [ ! -e "$R/.git" ]; then
    if [ -e "$R" ]; then echo "$R exists but is not a git checkout; resolve it by hand" >&2; exit 1; fi
    git clone -q -b main "$B" "$R" || exit 1
    {set_origin}fetch || exit 1
    say "cloned $R at $(at HEAD)"
else
    fetch || {{ echo "cannot fetch the console's bundle into $R" >&2; exit 1; }}
    branch="$(git -C "$R" symbolic-ref -q --short HEAD)"
    if [ "$(git -C "$R" rev-parse HEAD)" = "$(git -C "$R" rev-parse console/main)" ]; then
        say "$R is at console/main ($(at HEAD))"
    elif [ "$branch" != main ]; then
        say "$R is on ${{branch:-a detached HEAD}}; console/main ($(at console/main)) fetched, not merged"
    elif [ -n "$(git -C "$R" status --porcelain --untracked-files=no)" ]; then
        say "$R has uncommitted changes; console/main ($(at console/main)) fetched, not merged"
    elif git -C "$R" merge-base --is-ancestor HEAD console/main; then
        git -C "$R" merge -q --ff-only console/main && say "$R fast-forwarded to $(at HEAD)"             || say "$R could not fast-forward to console/main; left as is"
    else
        say "$R has commits of its own; console/main ($(at console/main)) fetched: git rebase console/main"
    fi
fi
rm -f "$B""#
    )
}

struct Steps {
    n: usize,
    total: usize,
}

impl Steps {
    fn next(&mut self, what: &str) {
        self.n += 1;
        println!("[{}/{}] {what}", self.n, self.total);
    }
}

/// A worker's artifact root given by hand: absolute, under /<year> (the only directory a
/// worker keeps: containers run --rm), no `..`.
fn check_artifact_root(root: &str, year: &str) -> Result<()> {
    let ok = root.starts_with(&format!("/{year}/"))
        && root.trim_end_matches('/').len() > year.len() + 1
        && !root.split('/').any(|c| c == "..")
        && !root.contains(['\n', '\t', '\r']);
    if !ok {
        bail!("--artifact-root must be a directory under /{year}/ (its only persistent mount), got '{root}'");
    }
    Ok(())
}

/// `docker container inspect --format` for [`Existing`]: the id, the image, the working
/// directory, then `mount <destination> <source>` and `env <NAME=value>` lines.
const EXISTING_FORMAT: &str = "{{.Id}}{{println}}{{.Config.Image}}{{println}}{{.Config.WorkingDir}}\
     {{range .Mounts}}{{println}}mount {{.Destination}} {{.Source}}{{end}}\
     {{range .Config.Env}}{{println}}env {{.}}{{end}}";

/// What an existing akao_<name> was created with; init adapts to it rather than to today.
#[derive(Debug, PartialEq)]
pub struct Existing {
    /// What step 6 checks it is still the same container by.
    pub id: String,
    pub image: String,
    /// Its $AKAO_ARTIFACT_ROOT, else (a container from before it) its working directory,
    /// where every `docker exec` starts.
    pub root: String,
    /// Whether the root is pinned ($AKAO_ARTIFACT_ROOT) rather than the working directory.
    pub pinned: bool,
    /// (destination, host source) of its bind mounts.
    pub mounts: Vec<(String, String)>,
}

impl Existing {
    fn parse(inspect: &str) -> Option<Existing> {
        let mut lines = inspect.lines().map(str::trim);
        let id = lines.next().filter(|i| !i.is_empty())?.to_string();
        let image = lines.next().filter(|i| !i.is_empty())?.to_string();
        let workdir = lines.next()?.to_string();
        let prefix = format!("env {}=", state::ARTIFACT_ROOT_ENV);
        let (mut pinned, mut mounts) = (None, Vec::new());
        for l in lines {
            if let Some(root) = l.strip_prefix(&prefix) {
                pinned = Some(root.to_string());
            } else if let Some((dst, src)) = l.strip_prefix("mount ").and_then(|m| m.split_once(' ')) {
                mounts.push((dst.to_string(), src.to_string()));
            }
        }
        Some(Existing {
            id,
            image,
            pinned: pinned.is_some(),
            root: pinned.unwrap_or(workdir),
            mounts,
        })
    }

    fn mount(&self, dst: &str) -> Option<&str> {
        self.mounts.iter().find(|(d, _)| d == dst).map(|(_, s)| s.as_str())
    }
}

/// Make the plan agree with an existing container: its image and artifact root stay what
/// they were.  An explicit request it cannot satisfy fails before anything changes.
fn adopt(p: &mut Plan, e: &Existing, opts: &Options) -> Result<()> {
    println!("  {} exists: image {}, artifact root {}", p.container, e.image, e.root);
    let remove = format!("docker --context {} rm -f {}", p.context, p.container);
    if e.image != p.image {
        if opts.image.is_some() {
            bail!(
                "{} runs {}, not the requested {}; pick another name, or remove it first ({remove})",
                p.container,
                e.image,
                p.image
            );
        }
        println!(
            "  it keeps its image; {} applies to a new container only ({remove})",
            p.image
        );
        p.image = e.image.clone();
    }
    let year = e.root.split('/').nth(1).unwrap_or_default();
    if year != p.year {
        bail!(
            "{} works in {} and mounts /{year}, not this year's /{}; give the new year's worker a new name",
            p.container,
            e.root,
            p.year
        );
    }
    // Init writes the control plane, the home and mirror's brief where hosts.tsv says; the
    // container must read them from there.
    for (dst, src) in [
        (format!("/{}", p.year), p.host_year_dir()),
        ("/root".to_string(), p.host_container_home.clone()),
    ] {
        match e.mount(&dst) {
            Some(have) if have.trim_end_matches('/') == src => {}
            have => bail!(
                "{} mounts {} at {dst}, but hosts.tsv now puts it at {src}; pick another name, or remove it first ({remove})",
                p.container,
                have.unwrap_or("nothing")
            ),
        }
    }
    // The brief and the task dirs are written under the /<year> mount; another mount at or
    // above the root would hide them from the container.
    if let Some((dst, src)) = e
        .mounts
        .iter()
        .find(|(d, _)| *d != format!("/{}", p.year) && (e.root == *d || e.root.starts_with(&format!("{d}/"))))
    {
        bail!(
            "{}'s artifact root {} lies under its own mount {dst} (from {src}), not under /{}; \
             akao cannot write into it",
            p.container,
            e.root,
            p.year
        );
    }
    if e.root != p.artifact_root {
        if opts.artifact_root.is_some() || opts.week.is_some() {
            bail!(
                "{} works in {}, not the requested {}; pick another name, or remove it first ({remove})",
                p.container,
                e.root,
                p.artifact_root
            );
        }
        if !e.pinned {
            println!(
                "  it predates {}: its working directory is its artifact root \
                 (to pin one: {remove}, then akao init; /root and /{} are bind mounts and survive)",
                state::ARTIFACT_ROOT_ENV,
                p.year
            );
        }
        p.artifact_root = e.root.clone();
    }
    Ok(())
}

/// Brings the worker up; returns the plan with the artifact root the container really has.
pub fn run(state: &State, r: &Runner, opts: &Options) -> Result<Plan> {
    let mut p = Plan::new(state, opts)?;
    let nick = &p.host.nick.clone();
    println!(
        "akao init: {} on {nick}{}\n  image {}\n  host home {}, artifact root {}",
        p.container,
        if r.dry_run { " [dry run]" } else { "" },
        p.image,
        p.host.host_home,
        p.artifact_root
    );
    // What the repo step ships; checked before anything changes.
    let repo = if opts.skip_setup {
        None
    } else {
        Some(state::shippable_repo()?)
    };
    let mut s = Steps { n: 0, total: 12 };

    s.next("resolve host and container");
    let cfg = r.query(&r.ssh_config_argv(nick))?;
    let hostname = cfg.lines().find_map(|l| l.strip_prefix("hostname ")).unwrap_or(nick);
    println!("  {nick} -> {hostname}");
    // Before anything changes on the host: docker must answer (through the endpoint: no
    // context yet), and an existing container decides the image and the artifact root.
    let endpoint = format!("ssh://{nick}");
    let docker_h = |args: &[&str]| -> Vec<String> {
        ["docker", "-H", &endpoint]
            .iter()
            .chain(args)
            .map(|s| s.to_string())
            .collect()
    };
    let version = r.query(&docker_h(&["version", "--format", "{{.Server.Version}}"]))?;
    println!("  docker {} reachable", version.trim());
    let filter = format!("name=^/?{}$", p.container);
    let names = r.query(&docker_h(&["ps", "-a", "--filter", &filter, "--format", "{{.Names}}"]))?;
    let existing = if names.lines().any(|n| n.trim() == p.container) {
        let text = r.query(&docker_h(&[
            "container",
            "inspect",
            "--format",
            EXISTING_FORMAT,
            &p.container,
        ]))?;
        let e = Existing::parse(&text)
            .with_context(|| format!("cannot read what {} was created with: {text:?}", p.container))?;
        adopt(&mut p, &e, opts)?;
        Some(e)
    } else {
        None
    };

    s.next("prepare host directories");
    let script = format!(
        "{ESCALATE}$S mkdir -p {} {}/container_home",
        q(&format!("{}{}", p.host.home(), p.artifact_root)),
        q(p.host.home())
    );
    r.run(&r.ssh_argv(nick, &script))?;

    s.next("deploy extra paths");
    if opts.skip_setup {
        println!("  skipped (--skip-setup)");
    } else {
        let src = state.require("deploy_src")?;
        let paths: Vec<String> = state
            .require("deploy_paths")?
            .split_whitespace()
            .map(String::from)
            .collect();
        for path in &paths {
            if !Path::new(&src).join(path).exists() {
                bail!("deploy path missing locally: {src}/{path}");
            }
        }
        // Anything beyond the repo the user ships to every box (none by default); the repo
        // itself goes only into the worker's clone (step 8), nothing of it under /<year>.
        if paths.is_empty() {
            println!("  nothing to deploy (deploy_paths is empty)");
        } else {
            let extract = format!("{ESCALATE}$S tar -xzf - -C {}", q(&p.host_year_dir()));
            r.pipe(
                &tar_create(Path::new(&src), &paths, DEPLOY_EXCLUDES, true),
                &r.ssh_argv(nick, &extract),
            )?;
        }
    }

    s.next("container home");
    let home = q(&p.host_container_home);
    if r.probe(&r.ssh_argv(nick, &format!("test -d {home}")))?.is_some() {
        println!("  reusing existing {}", p.host_container_home);
    } else {
        let template = state.home_template();
        if !template.is_dir() {
            bail!("home template missing: {}", template.display());
        }
        let extract = format!("{ESCALATE}$S mkdir -p {home} && $S tar -xzf - -C {home}");
        r.pipe(
            &tar_create(&template, &[".".into()], &[], false),
            &r.ssh_argv(nick, &extract),
        )?;
    }

    s.next("docker context");
    // docker runs plain `ssh` found through PATH and cannot take our -F; it relies on
    // the ~/.local/bin/ssh wrapper from the home template to read our ssh config (step 1
    // proved it reaches the box).
    let inspect: Vec<String> = [
        "docker",
        "context",
        "inspect",
        "--format",
        "{{.Endpoints.docker.Host}}",
        &p.context,
    ]
    .map(String::from)
    .into();
    let host_arg = format!("host={endpoint}");
    match r.probe(&inspect)? {
        Some(ep) if ep.trim() == endpoint => println!("  reusing context {}", p.context),
        Some(ep) => {
            println!("  context {} points at {}; updating", p.context, ep.trim());
            r.run(&["docker", "context", "update", &p.context, "--docker", &host_arg].map(String::from))?
        }
        None => r.run(&["docker", "context", "create", &p.context, "--docker", &host_arg].map(String::from))?,
    }
    r.run(&["docker", "context", "use", &p.context].map(String::from))?;

    s.next("container");
    // Still the container step 1 judged (another controller may have replaced it since).
    let now = r.probe(&p.docker(&[
        "container",
        "inspect",
        "--format",
        "{{.Id}} {{.State.Status}}",
        &p.container,
    ]))?;
    let (id, status) = match now.as_deref().map(str::trim).and_then(|s| s.split_once(' ')) {
        Some((id, status)) => (Some(id), Some(status)),
        None => (None, None),
    };
    if id != existing.as_ref().map(|e| e.id.as_str()) {
        bail!(
            "{} changed since step 1 (removed, created or replaced); run init again",
            p.container
        );
    }
    match status {
        Some("running") => println!("  reusing running {}", p.container),
        Some("created" | "exited") => r.run(&p.docker(&["start", &p.container]))?,
        Some(other) => bail!("{} exists but is {other}; resolve it by hand", p.container),
        None => r.run(&p.docker_run_argv()?)?,
    }

    s.next("install packages");
    if opts.skip_setup {
        println!("  skipped (--skip-setup)");
    } else {
        let have_all = APT_PACKAGES
            .iter()
            .map(|p| format!("command -v {} >/dev/null", if *p == "docker.io" { "docker" } else { p }))
            .collect::<Vec<_>>()
            .join(" && ");
        let script = format!(
            "{have_all} || {{ export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y {}; }}",
            APT_PACKAGES.join(" ")
        );
        r.run(&p.exec(&[], &script))?;
    }

    s.next("akao-workflow clone");
    match &repo {
        None => println!("  skipped (--skip-setup)"),
        Some(repo) => {
            // A bundle of the console's branches: no network, unpushed commits included,
            // and fetching it never touches the worker's own work.
            let origin = r
                .probe(&["git", "-C", repo, "remote", "get-url", "origin"].map(String::from))?
                .map(|o| o.trim().to_string());
            let bundle: Vec<String> = ["git", "-C", repo, "bundle", "create", "-", "--branches", "--tags"]
                .map(String::from)
                .into();
            let receive = p.exec(&["-i"], &format!("mkdir -p /root/.cache && cat >{}", q(REPO_BUNDLE)));
            r.pipe(&bundle, &receive)?;
            r.run(&p.exec(
                &[],
                &repo_sync_script(REPO_BUNDLE, state::WORKER_REPO, origin.as_deref()),
            ))?;
            // The static oaka the hook built: git-ignored, so it travels beside the bundle,
            // into the same place in the clone (the library is the clone's oaka/).
            let bin = format!("{}/oaka/bin", state::WORKER_REPO);
            let receive = p.exec(
                &["-i"],
                &format!(
                    "mkdir -p {bin} && cat >{bin}/oaka.tmp && chmod +x {bin}/oaka.tmp && mv {bin}/oaka.tmp {bin}/oaka"
                ),
            );
            r.pipe(&["cat".to_string(), format!("{repo}/oaka/bin/oaka")], &receive)?;
        }
    }

    s.next("install tools and agents");
    if opts.skip_setup {
        println!("  skipped (--skip-setup)");
    } else {
        // The install scripts come with the repo (step 8), like everything akao ships.
        let utils = format!("{}/utils", state::WORKER_REPO);
        r.run(&p.exec(&[], &format!("command -v gh >/dev/null || bash {utils}/install_gh.sh")))?;
        r.run(&p.exec(
            &[],
            &format!("test -x /root/.local/bin/claude || bash {utils}/agent.sh --yes"),
        ))?;
        // oaka's gsm8k client.  From PyPI on its own: never sglang[test], which pulls
        // PyPI sglang over the image's.  Its deps are pure Python (no torch/triton/sglang).
        // Not fatal: an image whose pip refuses (e.g. a mirrored vllm/ATOM one) still gets
        // its tmux and agent; oaka doctor then warns that gsm8k clients cannot run.
        r.run(&p.exec(
            &[],
            "command -v sgl-eval >/dev/null || python3 -m pip install sgl-eval \
             || echo 'warning: pip install sgl-eval failed; gsm8k clients will not run here'",
        ))?;
        // oaka came with the clone (step 8); put it on PATH for the worker agent.
        let oaka = format!("{}/oaka/bin/oaka", state::WORKER_REPO);
        r.run(&p.exec(
            &[],
            &format!(
                "if [ -x {oaka} ]; then ln -sf {oaka} /usr/local/bin/oaka; else echo 'no {oaka}; oaka not linked'; fi"
            ),
        ))?;
    }

    s.next("InferenceX checkout");
    // oaka's benchmark client runs InferenceX's own code, so every worker needs it.
    // Cloned once and never pulled: an existing checkout may carry local work.
    let infx = format!("/{}/nocopy/InferenceX", p.year);
    match r.probe(&p.exec(&[], &format!("git -C {infx} log -1 --format='%h %cs %s'")))? {
        Some(head) => println!("  reusing {infx} at {}", head.trim()),
        None => {
            let repo = state.require("infx_repo")?;
            let script = format!(
                "if [ -e {infx} ]; then echo '{infx} exists but is not a git checkout; resolve it by hand' >&2; exit 1; fi; \
                 git clone --filter=blob:none {} {infx}",
                q(&repo)
            );
            r.run(&p.exec(&[], &script))?
        }
    }

    s.next("tmux controller window");
    if r.probe(&p.exec(&[], "tmux has-session"))?.is_some() {
        println!("  tmux already running in {}; left as is", p.container);
    } else {
        r.run(&p.exec(
            &[],
            &format!("tmux new-session -d -n controller -c {}", q(&p.artifact_root)),
        ))?;
        r.run(&p.exec(&[], "tmux send-keys -t :controller claude Enter"))?;
    }

    s.next("worker doctor");
    // The worker's GPU arch and ROCm version (what plans' profiles are checked against)
    // and every prerequisite oaka's scripts need, now that setup is done: a container
    // from the wrong image, or a step that silently failed, shows here.  Reported, not
    // fatal: the container is up, and the report names each fix.
    // The clone's oaka; a container set up before the repo shipped has /<year>/oaka's.
    let oaka = format!("{}/oaka/bin/oaka", state::WORKER_REPO);
    let legacy = format!("/{}/oaka/bin/oaka", p.year);
    let script = format!("b={oaka}; [ -x \"$b\" ] || b={legacy}; \"$b\" doctor 2>&1; true");
    match r.probe(&p.exec(&[], &script))? {
        Some(out) => {
            for line in out.lines() {
                println!("  {line}");
            }
        }
        None => println!("  cannot run {oaka} in {}; skipped", p.container),
    }

    let attach = p.docker(&["exec", "-it", &p.container, "tmux", "attach"]);
    println!("done. attach with:\n  {}", show(&attach));
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Plan {
        Plan {
            host: Host {
                nick: "f19-11".into(),
                image: None,
                model_path: "/mnt/raid/models".into(),
                docker_sock: Some("/data/docker.sock".into()),
                host_home: "/root/akao".into(),
                rest: Some("--shm-size=64g -e 'FOO=a b'".into()),
            },
            context: "f19-11".into(),
            container: "akao_exp".into(),
            image: "img:tag".into(),
            year: "2026".into(),
            artifact_root: "/2026/ww41/exp".into(),
            host_container_home: "/root/akao/container_home/akao_exp".into(),
        }
    }

    #[test]
    fn docker_run_layout() {
        let argv = plan().docker_run_argv().unwrap();
        let s = argv.join(" ");
        assert!(s.starts_with("docker --context f19-11 run --rm -d --privileged"));
        assert!(s.contains("--name=akao_exp"));
        assert!(s.contains("-v /mnt/raid/models:/model"));
        assert!(s.contains("-v /data/docker.sock:/var/run/docker.sock"));
        assert!(s.contains("-v /root/akao/2026:/2026"));
        assert!(s.contains("-v /root/akao/container_home/akao_exp:/root"));
        assert!(s.contains("-w /2026/ww41/exp"));
        // rest args come after the skeleton but before the image.
        assert!(argv.ends_with(&["--shm-size=64g", "-e", "FOO=a b", "img:tag", "sleep", "infinity"].map(String::from)));
        assert!(argv.contains(&"PYTHONPATH=".to_string()));
    }

    fn opts() -> Options {
        Options {
            nick: "f19-11".into(),
            name: "exp".into(),
            week: None,
            artifact_root: None,
            image: None,
            skip_setup: false,
        }
    }

    #[test]
    fn artifact_root_is_pinned_and_checked() {
        let argv = plan().docker_run_argv().unwrap();
        let s = argv.join(" ");
        assert!(
            s.contains("-e AKAO_ARTIFACT_ROOT=/2026/ww41/exp -e AKAO_REPO_ROOT=/root/akao-workflow -w /2026/ww41/exp"),
            "{s}"
        );
        assert!(check_artifact_root("/2026/nocopy/exp", "2026").is_ok());
        for bad in ["/2026/", "/2026", "2026/x", "/2027/ww01/x", "/2026/../etc", "/tmp/x"] {
            assert!(check_artifact_root(bad, "2026").is_err(), "{bad}");
        }
    }

    /// `docker container inspect` output for a container plan() would reuse.
    fn inspect(image: &str, workdir: &str, root: Option<&str>) -> String {
        let mut t = format!(
            "0123abcd\n{image}\n{workdir}\nmount /model /mnt/raid/models\nmount /2026 /root/akao/2026\n\
             mount /root /root/akao/container_home/akao_exp\nenv PATH=/usr/bin\n"
        );
        if let Some(r) = root {
            t += &format!("env AKAO_ARTIFACT_ROOT={r}\n");
        }
        t
    }

    #[test]
    fn an_existing_container_keeps_its_image_and_root() {
        let e = Existing::parse(&inspect("img:old", "/2026/ww41/exp", Some("/2026/ww41/exp"))).unwrap();
        assert_eq!(
            (e.image.as_str(), e.root.as_str(), e.pinned),
            ("img:old", "/2026/ww41/exp", true)
        );
        assert_eq!(e.mount("/2026"), Some("/root/akao/2026"));
        let legacy = Existing::parse(&inspect("img:tag", "/2026/ww40/exp", None)).unwrap();
        assert_eq!((legacy.root.as_str(), legacy.pinned), ("/2026/ww40/exp", false));
        assert_eq!(e.id, "0123abcd");
        assert_eq!(Existing::parse(""), None);
        assert_eq!(
            Existing::parse("0123abcd\nimg:tag"),
            None,
            "truncated output is not a container"
        );

        // Derived root and host image: the container's win.
        let mut p = plan();
        adopt(&mut p, &legacy, &opts()).unwrap();
        assert_eq!(
            (p.image.as_str(), p.artifact_root.as_str()),
            ("img:tag", "/2026/ww40/exp")
        );
        let mut p = plan();
        adopt(&mut p, &e, &opts()).unwrap();
        assert_eq!(p.image, "img:old");
        // Explicit requests it cannot satisfy fail.
        let mirror = Options {
            image: Some("img:tag".into()),
            ..opts()
        };
        let err = adopt(&mut plan(), &e, &mirror).unwrap_err().to_string();
        assert!(err.contains("runs img:old, not the requested img:tag"), "{err}");
        let week = Options {
            week: Some("ww40".into()),
            ..opts()
        };
        let err = adopt(&mut plan(), &legacy, &week).unwrap_err().to_string();
        assert!(
            err.contains("works in /2026/ww40/exp, not the requested /2026/ww41/exp"),
            "{err}"
        );
        // Another year's container mounts another /<year>.
        let old = Existing::parse(
            &inspect("img:tag", "/2025/ww52/exp", None)
                .replace("mount /2026 /root/akao/2026", "mount /2025 /root/akao/2025"),
        )
        .unwrap();
        let err = adopt(&mut plan(), &old, &opts()).unwrap_err().to_string();
        assert!(err.contains("mounts /2025, not this year's /2026"), "{err}");
        // hosts.tsv moved the host home since the container was made.
        let moved = Existing::parse(&inspect("img:tag", "/2026/ww41/exp", None).replace("/root/akao/2026", "/h1/2026"))
            .unwrap();
        let err = adopt(&mut plan(), &moved, &opts()).unwrap_err().to_string();
        assert!(
            err.contains("mounts /h1/2026 at /2026, but hosts.tsv now puts it at /root/akao/2026"),
            "{err}"
        );
        // A mount of its own over the root: what akao writes under /<year> would be hidden.
        let nested =
            Existing::parse(&(inspect("img:tag", "/2026/ww41/exp", None) + "mount /2026/ww41 /other\n")).unwrap();
        let err = adopt(&mut plan(), &nested, &opts()).unwrap_err().to_string();
        assert!(
            err.contains("lies under its own mount /2026/ww41 (from /other)"),
            "{err}"
        );
    }

    #[test]
    fn the_repo_reaches_a_worker_without_touching_its_work() {
        let dir = std::env::temp_dir().join(format!("akao-reposync-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("console")).unwrap();
        let git = |cwd: &Path, args: &[&str]| {
            let out = std::process::Command::new("git")
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "init.defaultBranch=main",
                ])
                .args(args)
                .current_dir(cwd)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        let (console, clone) = (dir.join("console"), dir.join("clone"));
        let commit = |cwd: &Path, file: &str| {
            std::fs::write(cwd.join(file), file).unwrap();
            git(cwd, &["add", "-A"]);
            git(cwd, &["commit", "-q", "-m", file]);
        };
        git(&console, &["init", "-q"]);
        commit(&console, "c1");
        let bundle = dir.join("x.bundle");
        let sync = || {
            git(
                &console,
                &[
                    "bundle",
                    "create",
                    "-q",
                    bundle.to_str().unwrap(),
                    "--branches",
                    "--tags",
                ],
            );
            let script = repo_sync_script(
                bundle.to_str().unwrap(),
                clone.to_str().unwrap(),
                Some("https://example.com/r.git"),
            );
            let out = std::process::Command::new("bash")
                .arg("-c")
                .arg(&script)
                .output()
                .unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            assert!(!bundle.exists(), "the bundle is removed");
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        assert!(sync().contains("cloned"));
        assert_eq!(
            git(&clone, &["remote", "get-url", "origin"]),
            "https://example.com/r.git"
        );
        assert!(sync().contains("is at console/main"));
        commit(&console, "c2");
        assert!(sync().contains("fast-forwarded to"));
        assert!(clone.join("c2").exists());
        // The worker's own commit is never moved; console/main is fetched for a rebase.
        commit(&clone, "w1");
        commit(&console, "c3");
        let out = sync();
        assert!(out.contains("has commits of its own"), "{out}");
        assert!(clone.join("w1").exists() && !clone.join("c3").exists());
        git(&clone, &["rebase", "-q", "console/main"]);
        // Uncommitted changes, then another branch: fetched, not merged.
        commit(&console, "c4");
        std::fs::write(clone.join("c1"), "edited").unwrap();
        assert!(sync().contains("has uncommitted changes"));
        git(&clone, &["checkout", "-q", "--", "c1"]);
        git(&clone, &["checkout", "-q", "-b", "topic"]);
        assert!(sync().contains("is on topic"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tar_owner_and_excludes() {
        let argv = tar_create(Path::new("/src"), &["extra".into()], DEPLOY_EXCLUDES, true);
        assert_eq!(argv[..6], ["tar", "-C", "/src", "-h", "--owner=0", "--group=0"]);
        assert!(argv.contains(&"--exclude=.claude".to_string()));
        assert!(argv.ends_with(&["-czf".into(), "-".into(), "extra".into()]));
    }
}
