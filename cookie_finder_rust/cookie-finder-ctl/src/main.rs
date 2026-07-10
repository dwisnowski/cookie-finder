mod config;
mod control_loop;
mod gimbal;
mod input;
mod ipc;

use clap::{Parser, Subcommand};
use config::DEFAULT_SOCKET;
use control_loop::{run_control_loop, ControlState};
use gimbal::PanTiltGimbal;
use ipc::run_ipc_server;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "cookie-finder-ctl", version, about = "Cookie Finder gimbal + gamepad daemon")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run IPC daemon + control loop (default for production)
    Daemon {
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    /// Standalone gamepad → gimbal (no IPC)
    Run,
    /// Home gimbal via one-shot IPC call
    Home {
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cookie_finder_ctl=info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Home { socket } => {
            send_ipc(&socket, r#"{"cmd":"home"}"#).await?;
        }
        Commands::Run => {
            let gimbal = Arc::new(PanTiltGimbal::default());
            let state = Arc::new(ControlState {
                gimbal: Arc::clone(&gimbal),
                input_enabled: Arc::new(AtomicBool::new(true)),
            });
            tokio::select! {
                _ = run_control_loop(state) => {}
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("shutting down");
                    gimbal.cleanup();
                }
            }
        }
        Commands::Daemon { socket } => {
            let gimbal = Arc::new(PanTiltGimbal::default());
            let state = Arc::new(ControlState {
                gimbal: Arc::clone(&gimbal),
                input_enabled: Arc::new(AtomicBool::new(false)),
            });
            let loop_state = Arc::clone(&state);
            let socket_path = socket.clone();
            tokio::select! {
                r = run_ipc_server(&socket_path, state) => r?,
                _ = run_control_loop(loop_state) => {}
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("shutting down");
                    gimbal.cleanup();
                    let _ = std::fs::remove_file(&socket);
                }
            }
        }
    }
    Ok(())
}

async fn send_ipc(socket: &str, msg: &str) -> anyhow::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;
    let mut stream = UnixStream::connect(socket).await?;
    stream.write_all(msg.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    let mut buf = vec![0u8; 4096];
    let n = stream.read(&mut buf).await?;
    println!("{}", String::from_utf8_lossy(&buf[..n]));
    Ok(())
}
