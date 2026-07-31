use crate::config::{
    BT_CHANGE_THRESHOLD, BT_MOTION_MIN_ANGLE, BT_SENSITIVITY, LOOP_INTERVAL_MS,
};
use crate::gimbal::PanTiltGimbal;
use crate::input::GamepadInput;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const GAMEPAD_RETRY_INTERVAL: Duration = Duration::from_secs(2);

pub struct ControlState {
    pub gimbal: Arc<PanTiltGimbal>,
    pub input_enabled: Arc<AtomicBool>,
}

pub async fn run_control_loop(state: Arc<ControlState>) {
    let mut gamepad = match GamepadInput::open() {
        Ok(g) => Some(g),
        Err(e) => {
            tracing::warn!("gamepad unavailable at start: {e}");
            None
        }
    };

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

        if !state.input_enabled.load(Ordering::Relaxed) {
            continue;
        }

        // Hotplug: open (or reopen) the gamepad when input is enabled but no
        // device is available yet, or after the previous device vanished.
        if gamepad.is_none() && now.duration_since(last_gamepad_retry) >= GAMEPAD_RETRY_INTERVAL {
            last_gamepad_retry = now;
            match GamepadInput::open() {
                Ok(g) => {
                    tracing::info!("gamepad connected");
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
