use crate::config::VERSION;
use crate::control_loop::ControlState;
use crate::gimbal::PanTiltGimbal;
use anyhow::Context;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

#[derive(Debug, Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    pan: Option<f64>,
    #[serde(default)]
    tilt: Option<f64>,
    #[serde(default)]
    pan_hz: Option<f64>,
    #[serde(default)]
    tilt_hz: Option<f64>,
    #[serde(default)]
    direction: Option<i32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    enabled: Option<bool>,
}

fn ok_position(gimbal: &PanTiltGimbal) -> Value {
    let (pan, tilt) = gimbal.get_position();
    json!({"ok": true, "pan": pan, "tilt": tilt})
}

fn handle_request(req: &Request, state: &ControlState) -> Value {
    let g = &state.gimbal;
    match req.cmd.as_str() {
        "ping" => json!({"ok": true, "version": VERSION}),
        "get_position" => ok_position(g),
        "get_status" => {
            let (pan, tilt) = g.get_position();
            json!({
                "ok": true,
                "pan": pan,
                "tilt": tilt,
                "input_enabled": state.input_enabled.load(Ordering::Relaxed),
                "is_moving": g.is_moving(),
                "max_pan": g.max_pan,
                "max_tilt": g.max_tilt,
                "pan_hz": g.pan_hz(),
                "tilt_hz": g.tilt_hz(),
            })
        }
        "move_to_angles" => {
            let pan = req.pan.unwrap_or(0.0);
            let tilt = req.tilt.unwrap_or(0.0);
            g.move_to_angles(pan, tilt);
            ok_position(g)
        }
        "set_speed" => {
            let pan_hz = req.pan_hz.unwrap_or(500.0);
            let tilt_hz = req.tilt_hz.unwrap_or(500.0);
            g.set_speed(pan_hz, tilt_hz);
            json!({"ok": true})
        }
        "pan_step" => {
            let dir = req.direction.unwrap_or(1);
            let steps = req.steps.unwrap_or(1);
            g.pan_step(dir, steps);
            ok_position(g)
        }
        "tilt_step" => {
            let dir = req.direction.unwrap_or(1);
            let steps = req.steps.unwrap_or(1);
            g.tilt_step(dir, steps);
            ok_position(g)
        }
        "home" => {
            g.home();
            ok_position(g)
        }
        "stop" => {
            g.stop();
            json!({"ok": true})
        }
        "disable_motors" => {
            g.disable_motors();
            json!({"ok": true})
        }
        "set_input_enabled" => {
            let enabled = req.enabled.unwrap_or(false);
            state.input_enabled.store(enabled, Ordering::Relaxed);
            json!({"ok": true, "input_enabled": enabled})
        }
        other => json!({"ok": false, "error": format!("unknown cmd: {other}")}),
    }
}

async fn handle_client(mut stream: UnixStream, state: Arc<ControlState>) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.split();
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Request>(line) {
            Ok(req) => handle_request(&req, &state),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        };
        writer.write_all(resp.to_string().as_bytes()).await?;
        writer.write_all(b"\n").await?;
    }
    Ok(())
}

pub async fn run_ipc_server(socket_path: &str, state: Arc<ControlState>) -> anyhow::Result<()> {
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path).context("bind unix socket")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o666))?;
    }
    tracing::info!("IPC listening on {socket_path}");

    loop {
        let (stream, _) = listener.accept().await?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, state).await {
                tracing::debug!("ipc client: {e}");
            }
        });
    }
}
