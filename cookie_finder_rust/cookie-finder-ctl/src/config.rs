use anyhow::Context;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_SOCKET: &str = "/tmp/cookie-finder.sock";
#[cfg(target_os = "linux")]
pub const GPIO_CHIP: &str = "/dev/gpiochip1";

pub const PAN_CONTROL_PINS: [u32; 4] = [271, 268, 258, 272];
pub const TILT_CONTROL_PINS: [u32; 4] = [262, 229, 233, 265];

pub const MAX_PAN: f64 = 150.0;
pub const MAX_TILT: f64 = 60.0;
pub const DEFAULT_PAN_HZ: f64 = 500.0;
pub const DEFAULT_TILT_HZ: f64 = 500.0;
pub const STEPS_PER_REV: f64 = 4076.0;

pub const BT_DEADZONE: f64 = 0.15;
pub const BT_SENSITIVITY: f64 = 100.0;
pub const BT_CHANGE_THRESHOLD: f64 = 0.05;
pub const BT_MOTION_MIN_ANGLE: f64 = 0.01;
pub const LOOP_INTERVAL_MS: u64 = 5;

pub const DEFAULT_PHASE_ORDER: [usize; 4] = [0, 1, 2, 3];

#[derive(Debug, Clone)]
pub struct GimbalConfig {
    pub pan_phase_order: [usize; 4],
    pub tilt_phase_order: [usize; 4],
}

impl Default for GimbalConfig {
    fn default() -> Self {
        Self {
            pan_phase_order: DEFAULT_PHASE_ORDER,
            tilt_phase_order: DEFAULT_PHASE_ORDER,
        }
    }
}

#[derive(Debug, Deserialize)]
struct GimbalTomlSection {
    #[serde(default = "default_phase_order_vec")]
    pan_phase_order: Vec<usize>,
    #[serde(default = "default_phase_order_vec")]
    tilt_phase_order: Vec<usize>,
}

fn default_phase_order_vec() -> Vec<usize> {
    DEFAULT_PHASE_ORDER.to_vec()
}

#[derive(Debug, Deserialize)]
struct GimbalTomlRoot {
    #[serde(default)]
    gimbal: GimbalTomlSection,
}

impl Default for GimbalTomlSection {
    fn default() -> Self {
        Self {
            pan_phase_order: default_phase_order_vec(),
            tilt_phase_order: default_phase_order_vec(),
        }
    }
}

pub fn validate_phase_order(order: &[usize], name: &str) -> anyhow::Result<[usize; 4]> {
    if order.len() != 4 {
        anyhow::bail!("{name} must have exactly 4 elements, got {}", order.len());
    }
    let mut seen = [false; 4];
    let mut out = [0usize; 4];
    for (i, &v) in order.iter().enumerate() {
        if v > 3 {
            anyhow::bail!("{name}[{i}] = {v}: each value must be 0-3");
        }
        if seen[v] {
            anyhow::bail!("{name}: duplicate phase index {v}");
        }
        seen[v] = true;
        out[i] = v;
    }
    Ok(out)
}

pub fn resolve_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    if let Ok(env) = std::env::var("COOKIE_FINDER_CONFIG") {
        if !env.is_empty() {
            return Some(PathBuf::from(env));
        }
    }
    let local = PathBuf::from("config/gimbal.toml");
    if local.exists() {
        return Some(local);
    }
    let system = PathBuf::from("/etc/cookie-finder/gimbal.toml");
    if system.exists() {
        return Some(system);
    }
    None
}

pub fn load_gimbal_config(explicit: Option<&Path>) -> anyhow::Result<GimbalConfig> {
    let Some(path) = resolve_config_path(explicit) else {
        tracing::info!("no gimbal config file found, using default phase order");
        return Ok(GimbalConfig::default());
    };

    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read gimbal config {}", path.display()))?;
    let root: GimbalTomlRoot = toml::from_str(&text)
        .with_context(|| format!("parse gimbal config {}", path.display()))?;

    let config = GimbalConfig {
        pan_phase_order: validate_phase_order(&root.gimbal.pan_phase_order, "pan_phase_order")?,
        tilt_phase_order: validate_phase_order(&root.gimbal.tilt_phase_order, "tilt_phase_order")?,
    };
    tracing::info!(
        "loaded gimbal config from {} (pan={:?}, tilt={:?})",
        path.display(),
        config.pan_phase_order,
        config.tilt_phase_order
    );
    Ok(config)
}
