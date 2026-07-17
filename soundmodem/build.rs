//! Build the vendored HIT LilacSat soundmodem C core and our reentrant port.
//!
//! Two translation groups are compiled into one static library:
//!   * `vendor/` — upstream GPL C (FEC chain), built verbatim with warnings
//!     silenced; it is not held to our lint bar.
//!   * `csrc/`   — our reentrant demodulator and C ABI. These are verified
//!     separately under `clang -Weverything` (minus -Wunsafe-buffer-usage
//!     and -Wpadded, which are inapplicable to C); see the crate README.
//!
//! `_GNU_SOURCE` / `_USE_MATH_DEFINES` expose `M_SQRT2` on glibc / MSVC.

use std::env;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    let mut build = cc::Build::new();
    build
        .include("csrc")
        .include("vendor")
        .define("_GNU_SOURCE", None)
        .define("_USE_MATH_DEFINES", None)
        .warnings(false)
        .file("csrc/smdsp.c")
        .file("csrc/sm_api.c")
        .file("csrc/tm_decode.c")
        .file("csrc/tm_tables.c")
        .file("vendor/ccsds.c")
        .file("vendor/viterbi27.c")
        .file("vendor/rs.c")
        .file("vendor/randomizer.c")
        .file("vendor/metrics.c")
        .file("vendor/tab.c")
        .file("vendor/fifobuf.c")
        .compile("openhoshimi_soundmodem");

    // The C core uses libm (sinf/cosf/...); MSVC folds it into the CRT.
    if !target.contains("msvc") {
        println!("cargo:rustc-link-lib=m");
    }

    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=build.rs");
}
