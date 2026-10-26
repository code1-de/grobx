//! The Python extension module `grobx`, a thin wrapper over the core: the
//! one API every measurement goes through (`OBJECTIVES.md`, _Measurement_).

use std::ffi::CStr;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ptr;
use std::time::Duration;

use grobx_core::{CheckerMode, Error as CoreError, MonomialOrder, Params, Stop, Threads, VERSION};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};

// small-block-time.4 design -- the Python module sets the crate's allocator as its global allocator, not the crate setting it for every program that links it (not measured)
#[global_allocator]
static GLOBAL: grobx_core::Allocator = grobx_core::Allocator;

// f4-gfp.37 design -- every error of the core is raised as grobx.Error, a subclass of ValueError, not returned in a result's field (not measured)
create_exception!(grobx, Error, PyValueError);

/// A flag another thread raises to stop a call.
// f4-gfp.33 practice: releasing the GIL -- grobx.Cancel wraps the engine's flag, which the call polls with the GIL released, not a threading.Event polled through the GIL
#[pyclass(name = "Cancel", module = "grobx", frozen)]
struct Cancel(grobx_core::Cancel);

#[pymethods]
impl Cancel {
    #[new]
    fn new() -> Cancel {
        Cancel(grobx_core::Cancel::new())
    }

    fn set(&self) {
        self.0.set();
    }

    fn is_set(&self) -> bool {
        self.0.is_set()
    }
}

/// What a call returns.
#[pyclass(name = "Result", module = "grobx", frozen)]
struct Answer {
    /// The canonical basis text, None where a bound stopped the run.
    // f4-gfp.54 design -- the basis and cofactor texts become Python strings inside the call, and every read returns the same object, not a copy made at each read (not measured)
    #[pyo3(get)]
    basis: Option<Py<PyString>>,
    /// "deadline", "cancel", or "memory" where a bound stopped the run.
    #[pyo3(get)]
    stopped: Option<&'static str>,
    #[pyo3(get)]
    cofactors: Option<Py<PyString>>,
    #[pyo3(get)]
    certified: Option<bool>,
    #[pyo3(get)]
    flaw: Option<String>,
    /// The call report where the call asked for it, else None.
    // call-report.16 design -- Python takes a keyword report, False by default, and Result.report is a str made inside the call, or None, not a list of parsed lines nor a callback per line (not measured)
    #[pyo3(get)]
    report: Option<Py<PyString>>,
    /// The seed the call ran under, the one drawn where `seed` was None.
    #[pyo3(get)]
    seed: u64,
}

#[pymethods]
impl Answer {
    /// SHA-256 of the basis text, None where there is none.
    // f4-gfp.36 design -- the digest is computed when it is read, outside the timed call, not inside it (not measured)
    #[getter]
    fn digest(&self, py: Python<'_>) -> PyResult<Option<String>> {
        match &self.basis {
            None => Ok(None),
            Some(basis) => Ok(Some(grobx_core::digest(basis.bind(py).to_str()?))),
        }
    }
}

/// The reduced Gröbner basis of the instance `text` (`OBJECTIVES.md`,
/// _Measurement_ and _Call parameters_). `order` is "grevlex", "lex", or a
/// block order "block:k1,…,km", as the core's `MonomialOrder::from_str`
/// says. In lex the grevlex basis of a zero-dimensional ideal of degree at
/// most 65535 is changed to lex by FGLM, and otherwise F4 runs in lex. The
/// characteristic is a prime below 2^64, or 0 for Q, where the basis is
/// lifted from runs over primes of [2^61, 2^62), as the core's `groebner`
/// says.
/// `threads` 0 is all the machine has, `seed` None draws one
/// (`grobx_core::random_seed`), `deadline` is in milliseconds,
/// `memory` in bytes, and `checker` is "exact" or the number of points of
/// the cheap mode. `report` asks for the call report.
#[pyfunction]
#[pyo3(signature = (text, *, order="grevlex", threads=0, seed=None, deadline=None, cancel=None,
                    memory=None, checker=None, cofactors=false, report=false))]
#[allow(
    clippy::too_many_arguments,
    reason = "the keywords of grobx.groebner, which pyo3 takes as one argument each"
)]
fn groebner(
    py: Python<'_>,
    text: &str,
    order: &str,
    threads: usize,
    seed: Option<u64>,
    deadline: Option<u64>,
    cancel: Option<Bound<'_, Cancel>>,
    memory: Option<u64>,
    checker: Option<Bound<'_, PyAny>>,
    cofactors: bool,
    report: bool,
) -> PyResult<Answer> {
    // f4-gfp.31 design -- threads = 0 means all the machine has, not None (not measured)
    let threads = match NonZeroUsize::new(threads) {
        None => Threads::All,
        Some(n) => Threads::Count(n),
    };
    let order: MonomialOrder = order.parse().map_err(|e| raised(&e))?;
    // input-redundancy-time.10 design -- no keyword, switch, or field sets Params::echelon: the call runs with it on, as with signatures, not a keyword, a switch, and a field for it (not measured)
    let mut params = Params::new(order, threads, seed.unwrap_or_else(grobx_core::random_seed));
    params.deadline = deadline.map(Duration::from_millis);
    params.cancel = cancel.map(|c| c.get().0.clone());
    params.memory = memory;
    params.checker = checker.map(|c| checker_mode(&c)).transpose()?;
    params.cofactors = cofactors;
    params.report = report;
    // f4-gfp.33 practice: releasing the GIL -- the call releases the GIL while it computes, so another Python thread can raise the flag
    let outcome = py
        .detach(|| grobx_core::groebner(text, &params))
        .map_err(|e| raised(&e))?;
    // text-stop-latency.10 design -- the Python module's copy of each text into a Python string, and the C interface's handover of its texts, stay without a check, not a copy in chunks with a check between them (log 0057-1)
    Ok(Answer {
        basis: outcome.basis.map(|t| string_of(py, t)).transpose()?,
        stopped: outcome.stopped.map(Stop::name),
        cofactors: outcome.cofactors.map(|t| string_of(py, t)).transpose()?,
        certified: outcome.certified,
        flaw: outcome.flaw,
        report: outcome.report.map(|t| PyString::new(py, &t).unbind()),
        seed: params.seed,
    })
}

// text-copy-peak.3 tuned -- a text is handed to its Python string 2^18 bytes at a time, not 2^16, 2^20, or 2^21 bytes (log 0081-1)
/// The bytes `string_of` copies into a Python string at a time; a text of
/// at most this many is copied at once.
#[allow(
    clippy::expect_used,
    reason = "a constant: 2^18 is not zero, which the compiler evaluates"
)]
const TEXT_CHUNK: NonZeroUsize = NonZeroUsize::new(1 << 18).expect("2^18 is not zero");

// text-copy-peak.1 design -- a long text is copied into its Python string a chunk at a time by text::hand_over, which gives back the core's pages behind each chunk, not copied whole while the core's text stands, nor made a string at its first read after the call, nor written by the core into the string (log 0081-1)
// text-copy-peak.5 design -- a text of at most TEXT_CHUNK bytes is copied into its string at once, not handed over in chunks (not measured)
// text-copy-peak.8 design -- the basis and the cofactor texts become strings by string_of, and the report text by one copy, not the report text as well, whose CPU name may hold bytes past ASCII (not measured)
/// `text`, the core's ASCII, as a Python string. A text of at most
/// `TEXT_CHUNK` bytes is copied at once. A longer one is copied into a
/// string of its length a chunk at a time by `grobx_core::text::hand_over`,
/// which gives back the pages of `text` behind each chunk, so that the two
/// stand together about once; the string is then read back at its first,
/// middle, and last character.
fn string_of(py: Python<'_>, text: String) -> PyResult<Py<PyString>> {
    let len = text.len();
    if len <= TEXT_CHUNK.get() {
        return Ok(PyString::new(py, &text).unbind());
    }
    let ends = [0, len / 2, len - 1];
    let written = ends.map(|i| text.as_bytes()[i]);
    let (string, at) = blank(py, len)?;
    grobx_core::text::hand_over(text, TEXT_CHUNK, |from, part| {
        // SAFETY: `hand_over` hands chunks that cover [0, len) once, each
        // at its offset, and `at` points at the string's len characters,
        // which nothing else reads or writes before the string is
        // returned.
        unsafe { ptr::copy_nonoverlapping(part.as_ptr(), at.add(from), part.len()) };
    })?;
    // text-copy-peak.7 design -- the string is read back at its first, middle, and last character, not trusted unread, since the limited API does not say that PyUnicode_AsUTF8AndSize points at the string's characters (not measured)
    for (i, byte) in ends.into_iter().zip(written) {
        let index = ffi::Py_ssize_t::try_from(i)?;
        // SAFETY: `string` is a str of len characters, and index < len.
        let read = unsafe { ffi::PyUnicode_ReadChar(string.as_ptr(), index) };
        if read == ffi::Py_UCS4::MAX {
            return Err(PyErr::fetch(py));
        }
        if read != ffi::Py_UCS4::from(byte) {
            return Err(PyRuntimeError::new_err(format!(
                "the text's string reads {read} at {i}, where the text holds {byte}"
            )));
        }
    }
    Ok(string.unbind())
}

// text-copy-peak.2 design -- a string of len ASCII characters is made under the limited API by growing a string of two characters with PyUnicode_Resize, its characters reached by PyUnicode_AsUTF8AndSize, not by PyUnicode_New and PyUnicode_DATA outside the limited API, nor by a module built without abi3 for one interpreter (f4-gfp.32) (not measured)
/// The characters of the string `blank` grows: two, since the
/// interpreter may share a string of one character, which no resize may
/// change in place.
const SEED: &CStr = c"..";

/// A new Python string of `len` ASCII characters not yet written, `len`
/// at least 2, and a pointer to its characters, which nothing else holds.
fn blank(py: Python<'_>, len: usize) -> PyResult<(Bound<'_, PyString>, *mut u8)> {
    debug_assert!(len >= 2, "{len} characters");
    let size = ffi::Py_ssize_t::try_from(len)?;
    let seed = ffi::Py_ssize_t::try_from(SEED.count_bytes())?;
    // SAFETY: the string of `SEED` is new, so its one reference is this
    // function's, which PyUnicode_Resize may replace and gives back where
    // it fails; `raw` is then owned by `string`.
    let string = unsafe {
        let mut raw = ffi::PyUnicode_FromStringAndSize(SEED.as_ptr(), seed);
        if raw.is_null() {
            return Err(PyErr::fetch(py));
        }
        if ffi::PyUnicode_Resize(&raw mut raw, size) != 0 {
            ffi::Py_DecRef(raw);
            return Err(PyErr::fetch(py));
        }
        Bound::from_owned_ptr(py, raw).cast_into_unchecked::<PyString>()
    };
    let mut got: ffi::Py_ssize_t = 0;
    // SAFETY: `string` is a str, and `got` a Py_ssize_t to write.
    let at = unsafe { ffi::PyUnicode_AsUTF8AndSize(string.as_ptr(), &raw mut got) };
    if at.is_null() {
        return Err(PyErr::fetch(py));
    }
    if got != size {
        return Err(PyRuntimeError::new_err(format!(
            "a string of {len} characters reads {got} bytes"
        )));
    }
    Ok((string, at.cast_mut().cast::<u8>()))
}

// f4-gfp.34 design -- one keyword, "exact" or k >= 1 points of the cheap mode, not two keywords for the mode and the points (not measured)
fn checker_mode(value: &Bound<'_, PyAny>) -> PyResult<CheckerMode> {
    if let Ok(name) = value.cast::<PyString>() {
        return match name.to_str()? {
            "exact" => Ok(CheckerMode::Exact),
            other => Err(PyValueError::new_err(format!(
                "checker {other:?}: \"exact\" or a number of points"
            ))),
        };
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "checker: \"exact\" or a number of points, not a bool",
        ));
    }
    let points: u32 = value.extract()?;
    let points = NonZeroU32::new(points)
        .ok_or_else(|| PyValueError::new_err("checker: the cheap mode takes at least 1 point"))?;
    Ok(CheckerMode::Cheap { points })
}

/// The core's error as grobx.Error.
fn raised(e: &CoreError) -> PyErr {
    Error::new_err(format!("{variant}: {e}", variant = variant(e)))
}

/// The name of the error's variant, which the message of grobx.Error opens
/// with: "Error" for `Misfit`, which `groebner` never returns, for
/// `Malloc`, and for a variant the crate adds later.
fn variant(e: &CoreError) -> &'static str {
    match e {
        CoreError::Syntax { .. } => "Syntax",
        CoreError::UnknownVariable { .. } => "UnknownVariable",
        CoreError::RepeatedVariable { .. } => "RepeatedVariable",
        CoreError::Characteristic { .. } => "Characteristic",
        CoreError::Overflow { .. } => "Overflow",
        CoreError::Size => "Size",
        CoreError::NoInverse { .. } => "NoInverse",
        CoreError::Order { .. } => "Order",
        CoreError::Memory => "Memory",
        CoreError::Drops { .. } => "Drops",
        CoreError::Misfit { .. } | CoreError::Malloc { .. } | _ => "Error",
    }
}

// trace-replay.14 design -- the wrapper exposes neither grobx::learn nor grobx::replay, not keywords of grobx.groebner for the trace mode (not measured)
#[pymodule]
fn grobx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(groebner, m)?)?;
    m.add_class::<Cancel>()?;
    m.add_class::<Answer>()?;
    m.add("Error", m.py().get_type::<Error>())?;
    m.add("__version__", VERSION)?;
    Ok(())
}
