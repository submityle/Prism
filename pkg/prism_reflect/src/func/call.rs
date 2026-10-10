//! Function reflection: [`DynamicFunction`], [`FunctionInfo`], the
//! [`IntoFunction`] converter, and the typed [`FunctionError`].
//!
//! A reflected function erases a concrete `Fn($Args) -> Ret` into a uniform
//! `&[&dyn Reflect] -> Box<dyn Reflect>` call, recording the argument and
//! return type names in a [`FunctionInfo`]. Dispatch is **strict**: the arity
//! is checked before any argument is touched, and every argument is rebuilt
//! with [`FromReflect`] so a type mismatch is reported as a typed
//! [`FunctionError`] instead of being silently mis-read (design §12, §22 —
//! "错配会 UB" is prevented entirely in safe code).

use crate::from_reflect::FromReflect;
use crate::func::args::ArgList;
use crate::reflect::{Reflect, Typed};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use core::fmt;

/// Static signature of a reflected function: argument type names and the
/// return type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionInfo {
    arg_types: Vec<&'static str>,
    return_type: &'static str,
}

impl FunctionInfo {
    /// Build a signature from the ordered argument type names and the return
    /// type name.
    #[must_use]
    pub fn new(arg_types: Vec<&'static str>, return_type: &'static str) -> Self {
        Self {
            arg_types,
            return_type,
        }
    }

    /// The ordered argument type names.
    #[must_use]
    pub fn arg_types(&self) -> &[&'static str] {
        &self.arg_types
    }

    /// The number of declared arguments (the expected call arity).
    #[must_use]
    pub fn arg_count(&self) -> usize {
        self.arg_types.len()
    }

    /// The return type name.
    #[must_use]
    pub fn return_type(&self) -> &'static str {
        self.return_type
    }
}

/// An error produced while calling a reflected function.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionError {
    /// No function with the requested name is registered.
    UnknownFunction {
        /// The looked-up function name.
        name: String,
    },
    /// The number of supplied arguments does not match the signature.
    ArityMismatch {
        /// The called function's name, when known.
        function: Option<String>,
        /// The number of arguments the signature declares.
        expected: usize,
        /// The number of arguments actually supplied.
        actual: usize,
    },
    /// The argument at `index` is not of the type the signature expects.
    ArgTypeMismatch {
        /// The zero-based argument position.
        index: usize,
        /// The type name the signature expects.
        expected: &'static str,
        /// The runtime type name actually supplied.
        actual: &'static str,
    },
}

impl fmt::Display for FunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionError::UnknownFunction { name } => {
                write!(f, "no reflected function named `{name}` is registered")
            }
            FunctionError::ArityMismatch {
                function,
                expected,
                actual,
            } => match function {
                Some(name) => write!(f, "`{name}` expects {expected} argument(s), got {actual}"),
                None => write!(f, "function expects {expected} argument(s), got {actual}"),
            },
            FunctionError::ArgTypeMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "argument {index} type mismatch: expected `{expected}`, got `{actual}`"
            ),
        }
    }
}

impl std::error::Error for FunctionError {}

/// The type-erased call closure stored inside a [`DynamicFunction`].
///
/// It receives a slice whose length already matches the signature's arity
/// (checked by [`DynamicFunction::call`]) and returns the boxed result or a
/// per-argument [`FunctionError::ArgTypeMismatch`].
type ErasedCall =
    Box<dyn Fn(&[&dyn Reflect]) -> Result<Box<dyn Reflect>, FunctionError> + Send + Sync>;

/// A reflected, type-erased function callable by name through a
/// [`FunctionRegistry`](crate::FunctionRegistry).
///
/// Create one from any supported `Fn` with [`IntoFunction::into_function`], then
/// optionally name it with [`with_name`](DynamicFunction::with_name).
pub struct DynamicFunction {
    name: Option<String>,
    info: FunctionInfo,
    func: ErasedCall,
}

impl DynamicFunction {
    /// Assemble a function from its signature and erased call closure.
    pub(crate) fn from_parts(info: FunctionInfo, func: ErasedCall) -> Self {
        Self {
            name: None,
            info,
            func,
        }
    }

    /// Attach a name to this function, returning it for chaining.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The function's name, when one was assigned.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The function's static signature.
    #[must_use]
    pub fn info(&self) -> &FunctionInfo {
        &self.info
    }

    /// Call the function with `args`, validating arity and argument types.
    ///
    /// # Errors
    /// Returns [`FunctionError::ArityMismatch`] when the argument count differs
    /// from the signature, or [`FunctionError::ArgTypeMismatch`] when an
    /// argument's concrete type does not match the expected parameter type. No
    /// argument is read until both checks have passed.
    pub fn call(&self, args: &ArgList) -> Result<Box<dyn Reflect>, FunctionError> {
        let expected = self.info.arg_count();
        let actual = args.len();
        if expected != actual {
            return Err(FunctionError::ArityMismatch {
                function: self.name.clone(),
                expected,
                actual,
            });
        }
        (self.func)(&args.as_refs())
    }
}

impl fmt::Debug for DynamicFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicFunction")
            .field("name", &self.name)
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// Conversion from a concrete `Fn` into a [`DynamicFunction`].
///
/// `Marker` is an uninhabited-in-practice function-pointer type (`fn($Args) ->
/// Ret`) that lets each arity/signature have its own non-overlapping impl. It
/// is inferred at the call site, so callers write
/// `registry.register("name", some_fn)` without naming it.
///
/// Implemented for every `Fn` of arity 0 through 8 whose arguments implement
/// [`FromReflect`] + [`Typed`] and whose return type implements [`Reflect`].
pub trait IntoFunction<Marker> {
    /// Erase `self` into a [`DynamicFunction`].
    fn into_function(self) -> DynamicFunction;
}

impl<Func, Ret> IntoFunction<fn() -> Ret> for Func
where
    Func: Fn() -> Ret + Send + Sync + 'static,
    Ret: Reflect,
{
    fn into_function(self) -> DynamicFunction {
        let info = FunctionInfo::new(Vec::new(), type_name::<Ret>());
        let func: ErasedCall = Box::new(move |_args: &[&dyn Reflect]| {
            let result = (self)();
            Ok(Box::new(result) as Box<dyn Reflect>)
        });
        DynamicFunction::from_parts(info, func)
    }
}

/// Generate an [`IntoFunction`] impl for one fixed arity.
macro_rules! impl_into_function {
    ($(($Arg:ident, $val:ident, $idx:tt)),+ $(,)?) => {
        impl<Func, Ret, $($Arg,)+> IntoFunction<fn($($Arg,)+) -> Ret> for Func
        where
            Func: Fn($($Arg,)+) -> Ret + Send + Sync + 'static,
            $($Arg: FromReflect + Typed,)+
            Ret: Reflect,
        {
            fn into_function(self) -> DynamicFunction {
                let info = FunctionInfo::new(
                    alloc::vec![$(type_name::<$Arg>(),)+],
                    type_name::<Ret>(),
                );
                let func: ErasedCall = Box::new(move |args: &[&dyn Reflect]| {
                    $(
                        let $val = <$Arg as FromReflect>::from_reflect(args[$idx])
                            .ok_or_else(|| FunctionError::ArgTypeMismatch {
                                index: $idx,
                                expected: type_name::<$Arg>(),
                                actual: args[$idx].type_name(),
                            })?;
                    )+
                    let result = (self)($($val,)+);
                    Ok(Box::new(result) as Box<dyn Reflect>)
                });
                DynamicFunction::from_parts(info, func)
            }
        }
    };
}

impl_into_function!((A0, a0, 0));
impl_into_function!((A0, a0, 0), (A1, a1, 1));
impl_into_function!((A0, a0, 0), (A1, a1, 1), (A2, a2, 2));
impl_into_function!((A0, a0, 0), (A1, a1, 1), (A2, a2, 2), (A3, a3, 3));
impl_into_function!(
    (A0, a0, 0),
    (A1, a1, 1),
    (A2, a2, 2),
    (A3, a3, 3),
    (A4, a4, 4)
);
impl_into_function!(
    (A0, a0, 0),
    (A1, a1, 1),
    (A2, a2, 2),
    (A3, a3, 3),
    (A4, a4, 4),
    (A5, a5, 5)
);
impl_into_function!(
    (A0, a0, 0),
    (A1, a1, 1),
    (A2, a2, 2),
    (A3, a3, 3),
    (A4, a4, 4),
    (A5, a5, 5),
    (A6, a6, 6)
);
impl_into_function!(
    (A0, a0, 0),
    (A1, a1, 1),
    (A2, a2, 2),
    (A3, a3, 3),
    (A4, a4, 4),
    (A5, a5, 5),
    (A6, a6, 6),
    (A7, a7, 7)
);
