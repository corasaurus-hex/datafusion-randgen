#[test]
fn resolves_datafusion_53_1_0() {
    let lockfile = std::fs::read_to_string("Cargo.lock").expect("read Cargo.lock");
    let mut in_datafusion_package = false;

    for line in lockfile.lines() {
        match line {
            "name = \"datafusion\"" => in_datafusion_package = true,
            "version = \"53.1.0\"" if in_datafusion_package => return,
            line if line.starts_with("name = ") => in_datafusion_package = false,
            _ => {}
        }
    }

    panic!("Cargo.lock should resolve datafusion 53.1.0");
}
