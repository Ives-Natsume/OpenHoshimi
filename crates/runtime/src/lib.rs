//! Shared runtime plumbing for ClosedHoshimi decode pipelines.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

/// Analytic signal source (Hilbert transform wrapper).
pub mod analytic;

/// Offline block CCSDS frame recovery via ASM correlation.
pub mod block_demod;

/// Shared decode pipeline and runtime state.
pub mod pipeline;

/// IQ front-end feeding the vendored soundmodem C decoder (ASRTU-1 only).
pub mod soundmodem_frontend;
