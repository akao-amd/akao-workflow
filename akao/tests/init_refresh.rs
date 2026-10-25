//! Optional-image refresh preserves adoption, query failures and dry-run invariants.
mod common;
#[path = "common/init.rs"]
mod fixture;
use common::output;

#[test]
fn keep_infx_selects_verification_without_a_fetch_or_checkout() {
    let s = fixture::sandbox("keep-infx");
    let (code, out) = s.run(&[
        "init",
        "fakebox",
        "worker",
        "--image",
        "img:test",
        "--keep-infx",
        "--dry-run",
    ]);
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("--keep-infx requires an existing InferenceX checkout"),
        "{out}"
    );
    assert!(out.contains("kept InferenceX at %H %s (--keep-infx)"), "{out}");
    assert!(!out.contains("git clone -q --filter=blob:none"), "{out}");
    assert!(!out.contains("could not check out origin/main"), "{out}");
}

#[test]
fn refreshing_adopts_the_existing_image_and_pinned_or_legacy_root() {
    for pinned in [false, true] {
        let s = fixture::sandbox(if pinned { "refresh-pinned" } else { "refresh-legacy" });
        let root = format!("/{}/ww07/worker", fixture::year());
        let old = fixture::existing(&s, "img:old", &root, pinned);
        let (code, out) = output(
            s.command(&["init", "fakebox", "worker", "--dry-run"])
                .env("FAKE_EXISTING", &old),
        );
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("exists: image img:old"), "{out}");
        assert!(out.contains(&format!("artifact root {root}")), "{out}");
        assert!(out.contains("reusing running akao_worker"), "{out}");
        assert!(!s.events("docker.jsonl").iter().any(|a| a
            .iter()
            .any(|x| ["run", "start", "use", "create"].contains(&x.as_str()))));
    }
}

#[test]
fn missing_or_empty_images_fail_before_any_mutation() {
    let s = fixture::sandbox("refresh-missing");
    let (code, out) = s.run(&["init", "fakebox", "worker", "--dry-run"]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("does not exist on fakebox: give --image"), "{out}");
    assert!(s.events("docker.jsonl").iter().any(|a| a.contains(&"ps".into())));
    for image in ["", "  "] {
        let before = s.events("docker.jsonl").len();
        let (code, out) = s.run(&["init", "fakebox", "worker", "--image", image, "--dry-run"]);
        assert_ne!(code, 0, "{out}");
        assert!(out.contains("--image is empty"), "{out}");
        assert_eq!(s.events("docker.jsonl").len(), before);
    }
}

#[test]
fn refresh_rejects_conflicting_requests_and_a_replaced_container() {
    let s = fixture::sandbox("refresh-conflict");
    let root = format!("/{}/ww07/worker", fixture::year());
    let old = fixture::existing(&s, "img:old", &root, true);
    for extra in [
        vec!["--image", "img:new"],
        vec!["--week", "ww08"],
        vec!["--artifact-root", "/2026/elsewhere"],
    ] {
        let mut args = vec!["init", "fakebox", "worker", "--dry-run"];
        args.extend(extra);
        let (code, out) = output(s.command(&args).env("FAKE_EXISTING", &old));
        assert_ne!(code, 0, "{out}");
    }
    let (code, out) = output(
        s.command(&["init", "fakebox", "worker", "--dry-run"])
            .env("FAKE_EXISTING", &old)
            .env("FAKE_REPLACED", "new-id"),
    );
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("changed since step 1"), "{out}");
}

#[test]
fn unreadable_docker_or_inspection_is_never_reported_as_an_absent_worker() {
    for flag in ["FAKE_PS_DENIED", "FAKE_INSPECT_DENIED"] {
        let s = fixture::sandbox(flag);
        let root = format!("/{}/ww07/worker", fixture::year());
        let old = fixture::existing(&s, "img:old", &root, true);
        let (code, out) = output(
            s.command(&["init", "fakebox", "worker", "--dry-run"])
                .env("FAKE_EXISTING", old)
                .env(flag, "1"),
        );
        assert_ne!(code, 0, "{out}");
        assert!(!out.contains("does not exist"), "{out}");
    }
}
