use fearless_simd::Level;

use super::thread_pool::ThreadPool;

/// Holds long-lived context used for the effect (e.g. thread pool). Create one of these for the rough lifetime of your
/// application, or use the [`global`] one.
pub struct Context {
    pub(crate) thread_pool: ThreadPool,
    pub(crate) level: Level,
}

impl Context {
    pub fn new() -> Self {
        Self {
            thread_pool: ThreadPool::new(),
            #[cfg(feature = "std")]
            level: Level::new(),
            #[cfg(not(feature = "std"))]
            level: Level::baseline(),
        }
    }
}

#[cfg(feature = "std")]
static GLOBAL_CTX: std::sync::OnceLock<Context> = std::sync::OnceLock::new();

#[cfg(feature = "std")]
/// Global [`Context`], lazy-initialized on first use.
pub fn global() -> &'static Context {
    GLOBAL_CTX.get_or_init(Context::new)
}
