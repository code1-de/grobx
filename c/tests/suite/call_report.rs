//! call-report, sub-item 3: the report through the C interface. `report.c`
//! prints the header's sizes and the offsets of the two fields `report`,
//! which must be the library's, and calls `grobx_groebner` with the report
//! off, on, and on a call that fails (`iterations/0064-call-report/plan.md`,
//! _Test code_, 21, and call-report.17).

use std::collections::HashMap;
use std::fs;
use std::mem::{offset_of, size_of};

use grobx::{MonomialOrder, Params, Threads};
use grobxc::*;

use crate::common::{program, run, scratch, write_instance};
use crate::systems::dense;

/// Over GF(65521) and over ℚ.
#[test]
fn the_report_through_c() {
    for (stem, text) in [("p65521", dense(4, 3, 65521, 1)), ("q", dense(3, 3, 0, 1))] {
        let dir = scratch(&format!("call-report-{stem}"));
        let file = dir.join("report.txt");
        let out = run(
            &program("report.c"),
            &[
                write_instance(&dir, &text).display().to_string(),
                file.display().to_string(),
            ],
        );
        let stdout = String::from_utf8(out.stdout).unwrap();
        let lines: HashMap<&str, &str> = stdout
            .lines()
            .map(|l| {
                let split = if l.starts_with("sizeof") {
                    l.rsplit_once(' ')
                } else {
                    l.split_once(' ')
                };
                split.unwrap()
            })
            .collect();
        let library = [
            ("sizeof grobx_params", size_of::<grobx_params>()),
            ("grobx_params.report", offset_of!(grobx_params, report)),
            ("sizeof grobx_result", size_of::<grobx_result>()),
            ("grobx_result.report", offset_of!(grobx_result, report)),
        ];
        for (name, value) in library {
            assert_eq!(lines.get(name), Some(&value.to_string().as_str()), "{name}");
        }
        assert_eq!(
            lines["default"], "0",
            "grobx_params_new leaves the report off"
        );
        let ok = GROBX_OK.to_string();
        assert_eq!(
            lines["off"],
            format!("{ok} 1"),
            "{stem}: no report where none is asked"
        );

        let params = Params::new(
            MonomialOrder::Grevlex,
            Threads::Count(std::num::NonZeroUsize::MIN),
            11,
        );
        let basis = grobx::groebner(&text, &params)
            .unwrap()
            .basis
            .unwrap();
        assert_eq!(
            lines["on"],
            format!("{ok} 0 0 {bytes}", bytes = basis.len()),
            "{stem}: the report beside the basis"
        );
        assert_eq!(
            lines["freed"], "1 1",
            "{stem}: grobx_result_free sets both NULL"
        );
        assert_eq!(
            lines["error"],
            format!("{GROBX_OVERFLOW} 1"),
            "{stem}: an error returns no report"
        );

        let report = fs::read_to_string(&file).unwrap();
        let kinds: Vec<&str> = report
            .lines()
            .map(|l| l.split(' ').nth(1).expect("a kind"))
            .collect();
        assert_eq!(kinds.first(), Some(&"call"), "{stem}:\n{report}");
        assert_eq!(kinds.last(), Some(&"end"), "{stem}:\n{report}");
        let end = report.lines().last().unwrap();
        assert!(
            end.contains(" result=basis ")
                && end.contains(&format!(" bytes={bytes} ", bytes = basis.len())),
            "{stem}: {end}"
        );
    }
}
