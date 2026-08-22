use crate::config::{
    BT_CHANGE_THRESHOLD, BT_MOTION_MIN_ANGLE, BT_SENSITIVITY, LOOP_INTERVAL_MS,
};
use crate::gimbal::PanTiltGimbal;
use crate::input::{GamepadInput, GamepadSelector};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const GAMEPAD_RETRY_INTERVAL: Duration = Duration::from_secs(2);

pub struct ControlState {
    pub gimbal: Arc<PanTiltGimbal>,
    pub input_enabled: Arc<AtomicBool>,
    /// Desired gamepad (Bluetooth MAC / name). Changed from IPC without restart.
    pub active_input: Arc<Mutex<GamepadSelector>>,
    /// Bumped whenever enabled flag or active_input changes so the loop reopens.
    pub input_generation: Arc<AtomicU64>,
}

impl ControlState {
    pub fn bump_input_generation(&self) {
        self.input_generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_active_input(
        &self,
        enabled: bool,
        address: Option<&str>,
        name: Option<&str>,
    ) {
        let selector = if enabled {
            GamepadSelector::from_address_name(address, name)
        } else {
            GamepadSelector::any()
        };
        if let Ok(mut guard) = self.active_input.lock() {
            *guard = selector;
        }
        self.input_enabled.store(enabled, Ordering::Relaxed);
        self.bump_input_generation();
    }
}

pub async fn run_control_loop(state: Arc<ControlState>) {
    let mut applied_generation = u64::MAX;
    let mut gamepad: Option<GamepadInput> = None;
    let mut last_pan = 0.0f64;
    let mut last_tilt = 0.0f64;
    let mut last_tick = Instant::now();
    let mut last_gamepad_retry = Instant::now()
        .checked_sub(GAMEPAD_RETRY_INTERVAL)
        .unwrap_or_else(Instant::now);
    let interval = tokio::time::Duration::from_millis(LOOP_INTERVAL_MS);

    loop {
        tokio::time::sleep(interval).await;
        let now = Instant::now();
        let dt = now.duration_since(last_tick).as_secs_f64().max(0.001);
        last_tick = now;

        state.gimbal.tick();

        let generation = state.input_generation.load(Ordering::Relaxed);
        if generation != applied_generation {
            applied_generation = generation;
            if gamepad.is_some() {
                tracing::info!("active gamepad selection changed — releasing previous device");
            }
            gamepad = None;
            last_pan = 0.0;
            last_tilt = 0.0;
            // Allow immediate reopen attempt after a UI switch.
            last_gamepad_retry = Instant::now()
                .checked_sub(GAMEPAD_RETRY_INTERVAL)
                .unwrap_or_else(Instant::now);
        }

        if !state.input_enabled.load(Ordering::Relaxed) {
            continue;
        }

        let selector = state
            .active_input
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| GamepadSelector::any());

        // Hotplug / switch: open (or reopen) when enabled but no device yet.
        if gamepad.is_none() && now.duration_since(last_gamepad_retry) >= GAMEPAD_RETRY_INTERVAL {
            last_gamepad_retry = now;
            match GamepadInput::open(&selector) {
                Ok(g) => {
                    tracing::info!(
                        "gamepad connected: {} (address={:?} name={:?})",
                        g.path().display(),
                        selector.address,
                        selector.name
                    );
                    gamepad = Some(g);
                    last_pan = 0.0;
                    last_tilt = 0.0;
                }
                Err(e) => {
                    tracing::debug!("gamepad still unavailable: {e}");
                }
            }
        }

        let Some(gp) = gamepad.as_mut() else { continue };
        if gp.poll().is_err() {
            tracing::warn!("gamepad poll failed — will retry open");
            gamepad = None;
            last_gamepad_retry = Instant::now()
                .checked_sub(GAMEPAD_RETRY_INTERVAL)
                .unwrap_or_else(Instant::now);
            continue;
        }

        let (pan_axis, tilt_axis) = gp.axes();
        if (pan_axis - last_pan).abs() < BT_CHANGE_THRESHOLD
            && (tilt_axis - last_tilt).abs() < BT_CHANGE_THRESHOLD
        {
            continue;
        }
        last_pan = pan_axis;
        last_tilt = tilt_axis;

        let (cur_pan, cur_tilt) = state.gimbal.get_commanded();
        let new_pan = (cur_pan + pan_axis * BT_SENSITIVITY * dt)
            .clamp(0.0, state.gimbal.max_pan);
        let new_tilt = (cur_tilt + (-tilt_axis) * BT_SENSITIVITY * dt)
            .clamp(0.0, state.gimbal.max_tilt);

        if (new_pan - cur_pan).abs() < BT_MOTION_MIN_ANGLE
            && (new_tilt - cur_tilt).abs() < BT_MOTION_MIN_ANGLE
        {
            continue;
        }

        state.gimbal.move_to_angles(new_pan, new_tilt);
    }
}
