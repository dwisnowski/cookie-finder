const buttons = {
    'h': 'heat_seeker_mode',
    'c': 'heat_cluster_mode',
    'm': 'motion_mode',
    'p': 'palette_mode',
    't': 'threshold_mode',
    'y': 'yolo_mode',
    'f': 'optical_flow_mode',
    'i': 'isotherm_mode',
    'd': 'denoise_mode',
    'o': 'normalize_mode',
    'e': 'enhance_mode',
    'u': 'upscale_mode',
    's': 'stabilize_mode',
    'x': 'stabilize_super',
    'w': 'show_text'
};

const sliders = {
    'threshold_value': 'slider_threshold_value',
    'stabilize_strength': 'slider_stabilize_strength',
    'isotherm_min': 'slider_isotherm_min',
    'isotherm_max': 'slider_isotherm_max'
};

const palettes = [
    'Ironbow',
    'Rainbow',
    'Lava',
    'Ocean',
    'Magma',
    'WhiteHot',
    'BlackHot'
];

let ws = null;
let state = {};
let availableCameras = [];
let currentCamera = null;
let currentPaletteIdx = 0;

// Pan/Tilt tracking
const PAN_MAX = 150;
const TILT_MAX = 60;
const PAN_STEP = 5;
const TILT_STEP = 5;
const GAMEPAD_DEADZONE = 0.15;
const GAMEPAD_SENSITIVITY = 100;

let currentPan = 0;
let currentTilt = 0;
let motorActive = {};
let activeGamepadIndex = -1;
let connectedGamepads = [];
let lastGamepadPoll = Date.now();

let gamepadPanAxis = 0;
let gamepadTiltAxis = 1;
let gamepadInvertPan = false;
let gamepadInvertTilt = false;
const AXIS_NAMES = ['Left X', 'Left Y', 'Right X', 'Right Y'];

let currentPreset = 'normal';
const gamepadPresets = {
    'normal': {
        panAxis: 0,
        tiltAxis: 1,
        invertPan: false,
        invertTilt: false,
        label: 'Normal'
    },
    'vertical': {
        panAxis: 1,
        tiltAxis: 0,
        invertPan: true,
        invertTilt: false,
        label: 'Vertical'
    }
};

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function updateCameraSelector() {
    fetch('/available-cameras')
        .then(r => r.json())
        .then(data => {
            availableCameras = data.available;
            currentCamera = data.current;

            const selector = document.getElementById('cameraSelector');
            if (selector) {
                selector.innerHTML = '';

                if (availableCameras.length === 0) {
                    selector.innerHTML = '<p style="font-size: 11px; color: #aaa;">No cameras available</p>';
                } else {
                    availableCameras.forEach(cameraId => {
                        const btn = document.createElement('button');
                        btn.className = 'btn';
                        btn.style.width = '100%';
                        btn.style.marginBottom = '5px';
                        btn.textContent = `/dev/video${cameraId}`;

                        if (cameraId === currentCamera) {
                            btn.classList.add('active');
                        }

                        btn.addEventListener('click', () => {
                            switchCamera(cameraId);
                        });

                        selector.appendChild(btn);
                    });
                }
            }

            const currentCameraIdEl = document.getElementById('currentCameraId');
            if (currentCameraIdEl) {
                currentCameraIdEl.textContent = currentCamera !== null ? currentCamera : '--';
            }
        })
        .catch(e => console.error('Failed to fetch cameras:', e));
}

function updatePanTiltIndicator() {
    const svgRadius = 75;
    const panPercent = PAN_MAX === 0 ? 0 : clamp(currentPan / PAN_MAX, 0, 1);
    const tiltPercent = TILT_MAX === 0 ? 0 : clamp(currentTilt / TILT_MAX, 0, 1);

    const x = 100 + ((panPercent * 2) - 1) * svgRadius;
    const y = 100 - ((tiltPercent * 2) - 1) * svgRadius;

    const marker = document.getElementById('positionMarker');
    marker.setAttribute('cx', x);
    marker.setAttribute('cy', y);

    document.getElementById('markerLineH').setAttribute('x2', x);
    document.getElementById('markerLineH').setAttribute('y2', y);
    document.getElementById('markerLineV').setAttribute('x2', x);
    document.getElementById('markerLineV').setAttribute('y2', y);

    const panAngleEl = document.getElementById('panAngle');
    const tiltAngleEl = document.getElementById('tiltAngle');
    if (panAngleEl) panAngleEl.textContent = currentPan.toFixed(2) + '°';
    if (tiltAngleEl) tiltAngleEl.textContent = currentTilt.toFixed(2) + '°';
}

function updateGamepadStatus() {
    const gamepads = navigator.getGamepads?.() || [];
    connectedGamepads = Array.from(gamepads).filter(gp => gp !== null);

    const countDisplay = document.getElementById('gamepadDeviceCount');
    if (countDisplay) {
        countDisplay.textContent = connectedGamepads.length + ' device' + (connectedGamepads.length !== 1 ? 's' : '');
    }

    const indicator = document.getElementById('gamepadIndicator');
    const statusText = document.getElementById('gamepadStatusText');
    const nameDisplay = document.getElementById('gamepadNameDisplay');

    if (activeGamepadIndex >= 0 && activeGamepadIndex < connectedGamepads.length) {
        const activeGpad = connectedGamepads[activeGamepadIndex];
        if (indicator) indicator.classList.add('connected');
        if (statusText) statusText.textContent = '✓ Connected';
        if (nameDisplay) nameDisplay.textContent = activeGpad.id;
    } else if (connectedGamepads.length > 0) {
        activeGamepadIndex = 0;
        const activeGpad = connectedGamepads[0];
        if (indicator) indicator.classList.add('connected');
        if (statusText) statusText.textContent = '✓ Connected';
        if (nameDisplay) nameDisplay.textContent = activeGpad.id;
    } else {
        activeGamepadIndex = -1;
        if (indicator) indicator.classList.remove('connected');
        if (statusText) statusText.textContent = '✗ Disconnected';
        if (nameDisplay) nameDisplay.textContent = '—';
    }
}

function cycleGamepad() {
    if (connectedGamepads.length === 0) return;
    activeGamepadIndex = (activeGamepadIndex + 1) % connectedGamepads.length;
    updateGamepadStatus();
}

function pollGamepadInput() {
    if (activeGamepadIndex < 0 || activeGamepadIndex >= connectedGamepads.length) return;

    const gamepad = connectedGamepads[activeGamepadIndex];

    let panInput = gamepad.axes[gamepadPanAxis] || 0;
    let tiltInput = gamepad.axes[gamepadTiltAxis] || 0;

    if (gamepadInvertPan) panInput *= -1;
    if (gamepadInvertTilt) tiltInput *= -1;

    tiltInput *= -1;

    panInput = Math.abs(panInput) > GAMEPAD_DEADZONE ? panInput : 0;
    tiltInput = Math.abs(tiltInput) > GAMEPAD_DEADZONE ? tiltInput : 0;

    const timeDelta = (Date.now() - lastGamepadPoll) / 1000;
    lastGamepadPoll = Date.now();

    if (Math.abs(panInput) > 0.01 || Math.abs(tiltInput) > 0.01) {
        const panChange = panInput * GAMEPAD_SENSITIVITY * timeDelta;
        const tiltChange = tiltInput * GAMEPAD_SENSITIVITY * timeDelta;

        currentPan = clamp(currentPan + panChange, 0, PAN_MAX);
        currentTilt = clamp(currentTilt + tiltChange, 0, TILT_MAX);

        updatePanTiltIndicator();

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                action: 'motor_command',
                command: 'gamepad_input',
                pan: Math.round(currentPan),
                tilt: Math.round(currentTilt)
            }));
        }
    }

    updateGamepadButtonDisplay(gamepad);
}

function updateGamepadButtonDisplay(gamepad) {
    for (let i = 0; i < 4; i++) {
        const button = gamepad.buttons[i];
        const buttonElement = document.getElementById('gamepadButton' + i);

        if (button && button.pressed) {
            buttonElement.classList.add('pressed');
        } else {
            buttonElement.classList.remove('pressed');
        }
    }
}

function applyPreset(presetName) {
    if (!gamepadPresets[presetName]) return;

    const preset = gamepadPresets[presetName];
    currentPreset = presetName;
    gamepadPanAxis = preset.panAxis;
    gamepadTiltAxis = preset.tiltAxis;
    gamepadInvertPan = preset.invertPan;
    gamepadInvertTilt = preset.invertTilt;

    updateGamepadAxisDisplay();
}

function updateGamepadAxisDisplay() {
    for (let axisType of ['pan', 'tilt']) {
        const selectedAxis = axisType === 'pan' ? gamepadPanAxis : gamepadTiltAxis;
        for (let i = 0; i < 4; i++) {
            const btn = document.getElementById('btn_' + axisType + '_axis_' + i);
            if (i === selectedAxis) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }
    }

    const invertPanBtn = document.getElementById('btn_invert_pan');
    const invertTiltBtn = document.getElementById('btn_invert_tilt');

    if (gamepadInvertPan) {
        invertPanBtn.classList.add('active');
        invertPanBtn.textContent = 'Invert Pan: ON';
    } else {
        invertPanBtn.classList.remove('active');
        invertPanBtn.textContent = 'Invert Pan: OFF';
    }

    if (gamepadInvertTilt) {
        invertTiltBtn.classList.add('active');
        invertTiltBtn.textContent = 'Invert Tilt: ON';
    } else {
        invertTiltBtn.classList.remove('active');
        invertTiltBtn.textContent = 'Invert Tilt: OFF';
    }

    const normalBtn = document.getElementById('btn_preset_normal');
    const verticalBtn = document.getElementById('btn_preset_vertical');

    if (currentPreset === 'normal') {
        normalBtn.classList.add('active');
        verticalBtn.classList.remove('active');
    } else if (currentPreset === 'vertical') {
        verticalBtn.classList.add('active');
        normalBtn.classList.remove('active');
    } else {
        normalBtn.classList.remove('active');
        verticalBtn.classList.remove('active');
    }

    const presetDisplay = document.getElementById('currentPresetDisplay');
    let presetLabel = 'Custom';
    for (let key in gamepadPresets) {
        if (currentPreset === key) {
            presetLabel = gamepadPresets[key].label;
            break;
        }
    }
    presetDisplay.textContent = 'Current: ' + presetLabel;
}

function updateMotorAngle(command) {
    const increment = PAN_STEP;

    switch (command) {
        case 'motor_left':
            currentPan = clamp(currentPan - increment, 0, PAN_MAX);
            break;
        case 'motor_right':
            currentPan = clamp(currentPan + increment, 0, PAN_MAX);
            break;
        case 'motor_up':
            currentTilt = clamp(currentTilt + increment, 0, TILT_MAX);
            break;
        case 'motor_down':
            currentTilt = clamp(currentTilt - increment, 0, TILT_MAX);
            break;
        case 'motor_home':
            currentPan = 0;
            currentTilt = 0;
            break;
    }

    updatePanTiltIndicator();
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(protocol + '://' + window.location.host + '/control');

    ws.onopen = () => {
        document.getElementById('statusText').innerHTML = 'Connected';
        updateCameraSelector();
        // Fetch connected devices immediately and multiple times
        fetchConnectedDevices();
        setTimeout(fetchConnectedDevices, 500);
        setTimeout(fetchConnectedDevices, 1000);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
            state = msg.data;
            updateUI();
        } else if (msg.type === 'gimbal_position') {
            // Handle gimbal position updates from server (from BT device or web control)
            const pos = msg.data;
            currentPan = clamp(Number(pos.pan) || 0, 0, PAN_MAX);
            currentTilt = clamp(Number(pos.tilt) || 0, 0, TILT_MAX);
            updatePanTiltIndicator();
        } else if (msg.type === 'bluetooth') {
            // Handle Bluetooth status updates from controller
            const btUpdate = msg.data;
            if (btUpdate.status === 'scan_started') {
                bluetoothScanning = true;
                bluetoothDevices = [];
                updateBluetoothUI();
            } else if (btUpdate.status === 'scan_update') {
                bluetoothDevices = btUpdate.data.devices || [];
                updateBluetoothUI();
                // Refresh connected devices in case any newly scanned devices are connected
                fetchConnectedDevices();
            } else if (btUpdate.status === 'scan_complete') {
                bluetoothScanning = false;
                bluetoothDevices = btUpdate.data.devices || [];
                updateBluetoothUI();
                // Refresh connected devices after scan completes
                fetchConnectedDevices();
            } else if (btUpdate.status === 'device_connected' || btUpdate.status === 'device_disconnected' || btUpdate.status === 'device_removed') {
                // Refresh device list after connection/disconnection
                fetch('/bluetooth/devices')
                    .then(r => r.json())
                    .then(data => {
                        bluetoothDevices = data.devices;
                        bluetoothScanning = data.scanning;
                        updateBluetoothUI();
                        fetchConnectedDevices();
                    });
            }
        } else if (msg.type === 'bluetooth_state') {
            // Initial Bluetooth state on connection
            bluetoothDevices = msg.data.devices || [];
            bluetoothScanning = msg.data.scanning || false;
            updateBluetoothUI();
            fetchConnectedDevices();
        } else if (msg.type === 'bluetooth_scan_started') {
            bluetoothScanning = true;
            bluetoothDevices = [];
            updateBluetoothUI();
        } else if (msg.type === 'bluetooth_connect_result') {
            console.log('[BT] Connect result incoming:', msg);
            bluetoothConnectingDevices.delete(msg.address);
            updateBluetoothUI();
            fetchConnectedDevices();
            if (!msg.success) {
                const statusDisplay = document.getElementById('btStatusDisplay');
                if (statusDisplay) {
                    statusDisplay.textContent = `❌ Connection failed to ${msg.address}`;
                    statusDisplay.style.color = '#ff4444';
                    setTimeout(() => {
                        statusDisplay.style.color = '';
                        if (!bluetoothScanning) statusDisplay.textContent = 'Ready to scan';
                    }, 5000);
                }
            }
        } else if (msg.type === 'bluetooth_scan_stopped') {
            bluetoothScanning = false;
            updateBluetoothUI();
            fetchConnectedDevices();
        }
    };

    ws.onerror = (error) => {
        document.getElementById('statusText').innerHTML = 'Connection error';
    };

    ws.onclose = () => {
        setTimeout(connectWebSocket, 2000);
    };
}

function updateUI() {
    for (const [key, mode] of Object.entries(buttons)) {
        const btn = document.getElementById('btn_' + key);
        if (state[mode]) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }

    if (state.threshold_value !== undefined) {
        document.getElementById('slider_threshold_value').value = state.threshold_value;
        document.getElementById('val_threshold').textContent = state.threshold_value;
    }
    if (state.stabilize_strength !== undefined) {
        document.getElementById('slider_stabilize_strength').value = state.stabilize_strength * 100;
        document.getElementById('stab_strength').textContent = state.stabilize_strength.toFixed(1);
    }
    if (state.isotherm_min !== undefined) {
        document.getElementById('slider_isotherm_min').value = state.isotherm_min;
        document.getElementById('val_isotherm_min').textContent = state.isotherm_min;
    }
    if (state.isotherm_max !== undefined) {
        document.getElementById('slider_isotherm_max').value = state.isotherm_max;
        document.getElementById('val_isotherm_max').textContent = state.isotherm_max;
    }

    if (state.palette_idx !== undefined) {
        currentPaletteIdx = state.palette_idx;
        const paletteName = palettes[currentPaletteIdx] || 'Unknown';
        document.getElementById('currentPaletteName').textContent = paletteName;
    }

    const buttons_list = document.querySelectorAll('#cameraSelector button');
    buttons_list.forEach(btn => {
        const btnId = parseInt(btn.textContent.match(/\d+/)[0]);
        if (btnId === currentCamera) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    let statusLines = [];
    if (state.palette_name) statusLines.push('Palette: ' + state.palette_name);
    statusLines.push('Connected');
    document.getElementById('statusText').innerHTML = statusLines.join('<br>');

    // Update palette panel visibility
    updatePalettePanelVisibility();
    // Update parameters panel visibility
    updateParametersPanelVisibility();
    // Update status badges
    updateStatusBadges();
}

function updatePalettePanelVisibility() {
    const palettePanel = document.getElementById('palettePanel');
    if (state.palette_mode) {
        palettePanel.classList.add('active');
    } else {
        palettePanel.classList.remove('active');
    }
}

function updateParametersPanelVisibility() {
    const parametersPanel = document.getElementById('parametersPanel');
    if (state.isotherm_mode) {
        parametersPanel.classList.add('active');
    } else {
        parametersPanel.classList.remove('active');
    }
}

function updateStatusBadges() {
    // Update gamepad badge
    const gamepadDot = document.getElementById('badge_gamepad_dot');
    const gamepadText = document.getElementById('badge_gamepad_text');
    if (connectedGamepads && connectedGamepads.length > 0) {
        gamepadDot.classList.add('active');
        const name = connectedGamepads[activeGamepadIndex]?.id || 'Connected';
        gamepadText.textContent = name.substring(0, 20); // Truncate long names
    } else {
        gamepadDot.classList.remove('active');
        gamepadText.textContent = 'bluetooth';
    }

    // Update camera badge
    const cameraDot = document.getElementById('badge_camera_dot');
    const cameraText = document.getElementById('badge_camera_text');
    if (currentCamera !== null && currentCamera !== undefined) {
        cameraDot.classList.add('active');
        cameraText.textContent = `/dev/video${currentCamera}`;
    } else {
        cameraDot.classList.remove('active');
        cameraText.textContent = 'camera';
    }

    // FPS badge is updated by polling, so we'll just ensure the element exists
    const fpsDot = document.getElementById('badge_fps_dot');
    const fpsText = document.getElementById('badge_fps_text');
    if (fpsDot && fpsText) {
        fpsDot.classList.add('active');
        fpsText.textContent = '50 Hz';
    }
}


function switchCamera(newCameraId) {
    fetch(`/switch-camera/${newCameraId}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            currentCamera = newCameraId;
            updateCameraSelector();
            setTimeout(updateUI, 100);
        })
        .catch(e => console.error('Switch error:', e));
}

// Button handlers
for (const [key, mode] of Object.entries(buttons)) {
    document.getElementById('btn_' + key).addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                action: 'toggle_mode',
                mode: mode
            }));
        }
    });
}

// Slider handlers
document.getElementById('slider_threshold_value').addEventListener('change', (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'threshold_value',
            value: parseInt(e.target.value)
        }));
    }
});

document.getElementById('slider_stabilize_strength').addEventListener('change', (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'stabilize_strength',
            value: parseFloat(e.target.value) / 100
        }));
    }
});

document.getElementById('slider_isotherm_min').addEventListener('change', (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'isotherm_min',
            value: parseInt(e.target.value)
        }));
    }
});

document.getElementById('slider_isotherm_max').addEventListener('change', (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'isotherm_max',
            value: parseInt(e.target.value)
        }));
    }
});

// Palette cycling
document.getElementById('btn_palette_prev').addEventListener('click', () => {
    const newIdx = (currentPaletteIdx - 1 + palettes.length) % palettes.length;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'palette_idx',
            value: newIdx
        }));
    }
});

document.getElementById('btn_palette_next').addEventListener('click', () => {
    const newIdx = (currentPaletteIdx + 1) % palettes.length;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'palette_idx',
            value: newIdx
        }));
    }
});

// Motor control
const motorCommands = {
    'btn_motor_up': 'motor_up',
    'btn_motor_down': 'motor_down',
    'btn_motor_left': 'motor_left',
    'btn_motor_right': 'motor_right',
    'btn_motor_home': 'motor_home'
};

for (const [btnId, command] of Object.entries(motorCommands)) {
    const btn = document.getElementById(btnId);
    if (btn) {
        const startMotor = () => {
            motorActive[command] = true;

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: 'motor_command',
                    command: command,
                    state: 'start'
                }));
            }
        };

        const stopMotor = () => {
            motorActive[command] = false;

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: 'motor_command',
                    command: command,
                    state: 'stop'
                }));
            }
        };

        btn.addEventListener('mousedown', startMotor);
        btn.addEventListener('mouseup', stopMotor);
        btn.addEventListener('mouseleave', stopMotor);

        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startMotor();
        });

        btn.addEventListener('touchend', (e) => {
            e.preventDefault();
            stopMotor();
        });
    }
}

// Keyboard shortcuts
const keyPressState = {};

document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();

    if (buttons[key] && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'toggle_mode',
            mode: buttons[key]
        }));
    }

    if (key === '[' && ws && ws.readyState === WebSocket.OPEN) {
        const newIdx = (currentPaletteIdx - 1 + palettes.length) % palettes.length;
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'palette_idx',
            value: newIdx
        }));
        e.preventDefault();
    }
    if (key === ']' && ws && ws.readyState === WebSocket.OPEN) {
        const newIdx = (currentPaletteIdx + 1) % palettes.length;
        ws.send(JSON.stringify({
            action: 'set_param',
            param: 'palette_idx',
            value: newIdx
        }));
        e.preventDefault();
    }

    if ((key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') && ws && ws.readyState === WebSocket.OPEN) {
        const motorMap = {
            'arrowup': 'motor_up',
            'arrowdown': 'motor_down',
            'arrowleft': 'motor_left',
            'arrowright': 'motor_right'
        };

        if (!keyPressState[key]) {
            keyPressState[key] = true;
            const command = motorMap[key];
            motorActive[command] = true;

            ws.send(JSON.stringify({
                action: 'motor_command',
                command: command,
                state: 'start'
            }));
        }
        e.preventDefault();
    }
});

document.addEventListener('keyup', (e) => {
    const key = e.key.toLowerCase();

    if ((key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') && ws && ws.readyState === WebSocket.OPEN) {
        const motorMap = {
            'arrowup': 'motor_up',
            'arrowdown': 'motor_down',
            'arrowleft': 'motor_left',
            'arrowright': 'motor_right'
        };

        const command = motorMap[key];
        motorActive[command] = false;
        keyPressState[key] = false;

        ws.send(JSON.stringify({
            action: 'motor_command',
            command: command,
            state: 'stop'
        }));
        e.preventDefault();
    }

    if ((key === '?' || e.key === '?') || (e.shiftKey && key === '/')) {
        const isOpen = helpOverlay.classList.contains('active');
        if (isOpen) {
            closeHelp();
        } else {
            openHelp();
        }
        e.preventDefault();
    }
});

// Camera status polling
function updateCameraStatus() {
    fetch('/camera-status')
        .then(r => r.json())
        .then(data => {
            const indicator = document.getElementById('statusIndicator');
            const message = document.getElementById('statusMessage');

            if (indicator && message) {
                if (data.connected) {
                    indicator.style.background = '#00aa00';
                    message.textContent = '✓ Camera Connected';
                } else {
                    indicator.style.background = '#ff4444';
                    message.textContent = '✗ Camera Disconnected';
                }
            }
        })
        .catch(e => console.error('Status fetch error:', e));
}

// Settings modal
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsBtn = document.getElementById('btn_settings');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');

function openSettings() {
    settingsOverlay.classList.add('active');
}

function closeSettings() {
    settingsOverlay.classList.remove('active');
}

settingsBtn.addEventListener('click', openSettings);
closeSettingsBtn.addEventListener('click', closeSettings);

settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) {
        closeSettings();
    }
});

// Help modal
const helpOverlay = document.getElementById('helpOverlay');
const helpBtn = document.getElementById('btn_help');
const closeHelpBtn = document.getElementById('closeHelpBtn');

function openHelp() {
    helpOverlay.classList.add('active');
}

function closeHelp() {
    helpOverlay.classList.remove('active');
}

helpBtn.addEventListener('click', openHelp);
closeHelpBtn.addEventListener('click', closeHelp);

helpOverlay.addEventListener('click', (e) => {
    if (e.target === helpOverlay) {
        closeHelp();
    }
});

// Polling
setInterval(updateCameraStatus, 1000);
setInterval(updateCameraSelector, 3000);

updatePanTiltIndicator();

// Cycle gamepad button in settings modal
const cycleBtnSettings = document.getElementById('btn_cycle_gamepad_settings');
if (cycleBtnSettings) {
    cycleBtnSettings.addEventListener('click', cycleGamepad);
}

// Reconnect button in settings modal
const reconnectBtnSettings = document.getElementById('btn_reconnect_settings');
if (reconnectBtnSettings) {
    reconnectBtnSettings.addEventListener('click', () => {
        fetch('/reconnect', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                for (let i = 0; i < 10; i++) {
                    setTimeout(updateCameraStatus, i * 500);
                }
            })
            .catch(e => console.error('Reconnect error:', e));
    });
}

window.addEventListener('gamepadconnected', (e) => {
    updateGamepadStatus();
});

window.addEventListener('gamepaddisconnected', (e) => {
    updateGamepadStatus();
});

setInterval(() => {
    updateGamepadStatus();
    pollGamepadInput();
}, 50);

updateGamepadStatus();
updateGamepadAxisDisplay();

for (let i = 0; i < 4; i++) {
    const btn = document.getElementById('btn_pan_axis_' + i);
    if (btn) {
        btn.addEventListener('click', () => {
            gamepadPanAxis = i;
            updateGamepadAxisDisplay();
        });
    }
}

for (let i = 0; i < 4; i++) {
    const btn = document.getElementById('btn_tilt_axis_' + i);
    if (btn) {
        btn.addEventListener('click', () => {
            gamepadTiltAxis = i;
            updateGamepadAxisDisplay();
        });
    }
}

const invertPanBtn = document.getElementById('btn_invert_pan');
if (invertPanBtn) {
    invertPanBtn.addEventListener('click', () => {
        gamepadInvertPan = !gamepadInvertPan;
        updateGamepadAxisDisplay();
    });
}

const invertTiltBtn = document.getElementById('btn_invert_tilt');
if (invertTiltBtn) {
    invertTiltBtn.addEventListener('click', () => {
        gamepadInvertTilt = !gamepadInvertTilt;
        updateGamepadAxisDisplay();
    });
}

const normalPresetBtn = document.getElementById('btn_preset_normal');
if (normalPresetBtn) {
    normalPresetBtn.addEventListener('click', () => {
        applyPreset('normal');
    });
}

const verticalPresetBtn = document.getElementById('btn_preset_vertical');
if (verticalPresetBtn) {
    verticalPresetBtn.addEventListener('click', () => {
        applyPreset('vertical');
    });
}

// === BLUETOOTH FUNCTIONALITY ===
let bluetoothDevices = [];
let bluetoothScanning = false;
let hideUnknownDevices = true; // Default: hide Unknown Device
let bluetoothConnectedDevices = [];
let bluetoothConnectingDevices = new Set(); // Track devices being connected

function updateBluetoothUI() {
    const devicesList = document.getElementById('btDevicesList');
    const statusDisplay = document.getElementById('btStatusDisplay');
    const scanBtn = document.getElementById('btn_bt_scan');
    const stopBtn = document.getElementById('btn_bt_stop');

    if (bluetoothScanning) {
        scanBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        statusDisplay.textContent = 'Scanning for devices...';
    } else {
        scanBtn.style.display = 'block';
        stopBtn.style.display = 'none';
    }

    // Filter devices if 'Hide Unknown' is checked
    let displayDevices = bluetoothDevices;
    if (hideUnknownDevices) {
        displayDevices = displayDevices.filter(d => d.name && !d.name.includes('Unknown'));
    }

    if (displayDevices.length === 0) {
        if (bluetoothDevices.length > 0 && hideUnknownDevices) {
            devicesList.innerHTML = `<div style="color: var(--text-tertiary); font-size: 11px; text-align: center; padding: var(--spacing-md);">
                ${bluetoothDevices.length} unknown devices hidden.<br>
                <a href="#" onclick="document.getElementById('chk_hide_unknown').click(); return false;" style="color: var(--accent); text-decoration: underline;">Show them</a>
             </div>`;
        } else {
            devicesList.innerHTML = '<div style="color: var(--text-tertiary); font-size: 11px; text-align: center; padding: var(--spacing-md);">No devices found. Click "Scan Devices" to start.</div>';
        }

        if (!bluetoothScanning) {
            statusDisplay.textContent = 'Ready to scan';
        }
        return;
    }

    statusDisplay.textContent = `Found ${displayDevices.length} device${displayDevices.length !== 1 ? 's' : ''}`;
    if (hideUnknownDevices && bluetoothDevices.length > displayDevices.length) {
        statusDisplay.textContent += ` (${bluetoothDevices.length - displayDevices.length} hidden)`;
    }

    devicesList.innerHTML = displayDevices.map(device => {
        const addressShort = device.address.substring(device.address.length - 5).toUpperCase();
        const statusClass = device.connected ? 'connected' : device.paired ? 'paired' : '';
        const statusText = device.connected ? '🔗 Connected' : device.paired ? '✓ Paired' : '⊗ Available';
        const isConnecting = bluetoothConnectingDevices.has(device.address);

        return `
            <div style="
                padding: var(--spacing-sm);
                margin-bottom: var(--spacing-xs);
                background: rgba(45, 55, 72, 0.6);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-sm);
                font-size: 11px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-weight: 600; color: var(--text-primary);">
                        ${device.name || 'Unknown Device'}
                    </div>
                    <div style="font-size: 9px; color: var(--text-tertiary);">
                        ${addressShort} ${device.signal_strength}
                    </div>
                </div>
                <div style="margin-bottom: 4px; font-size: 10px; color: var(--accent);">
                    ${isConnecting ? '<span style="animation: blink 1s infinite;">🔄 Connecting...</span>' : statusText}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3px;">
                    ${!device.connected ? `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px; ${isConnecting ? 'opacity: 0.6; cursor: not-allowed;' : ''}" id="btn_connect_${device.address}" onclick="bluetoothConnect('${device.address}')" ${isConnecting ? 'disabled' : ''}>${isConnecting ? '⏳ Connecting...' : 'Connect'}</button>` : `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothDisconnect('${device.address}')">Disconnect</button>`}
                    <button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothSetActive('${device.address}')">Set Active</button>
                    <button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;"  onclick="bluetoothRemove('${device.address}')">Remove</button>
                </div>
            </div>
        `;
    }).join('');
}

function updateConnectedBluetoothUI() {
    const connectedList = document.getElementById('btConnectedList');

    if (!bluetoothConnectedDevices || bluetoothConnectedDevices.length === 0) {
        connectedList.innerHTML = '<div style="color: var(--text-tertiary); font-size: 11px; text-align: center; padding: var(--spacing-md);">No devices connected</div>';
        return;
    }

    connectedList.innerHTML = bluetoothConnectedDevices.map(device => {
        const addressShort = device.address.substring(device.address.length - 5).toUpperCase();
        const signalBar = device.signal_strength || '▓░░░░';
        const isActive = device.is_active;

        return `
            <div style="
                padding: var(--spacing-sm);
                margin-bottom: var(--spacing-xs);
                background: ${isActive ? 'rgba(0, 255, 136, 0.15)' : 'rgba(0, 255, 136, 0.08)'};
                border: 2px solid ${isActive ? 'rgba(0, 255, 136, 0.5)' : 'rgba(0, 255, 136, 0.3)'};
                border-radius: var(--radius-sm);
                font-size: 11px;
                position: relative;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-weight: 600; color: var(--accent); display: flex; align-items: center; gap: 6px;">
                        🔗 ${device.name || 'Unknown Device'}
                        ${isActive ? '⭐' : ''}
                    </div>
                    <div style="font-size: 9px; color: var(--text-tertiary);">
                        ${addressShort} ${signalBar}
                    </div>
                </div>
                ${isActive ? '<div style="font-size: 9px; color: var(--accent); margin-bottom: 4px; font-weight: bold;">● ACTIVE INPUT</div>' : ''}
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3px;">
                    ${!isActive ? `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothSetActive('${device.address}')">Set Active</button>` : '<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px; opacity: 0.5; cursor: default;">Active</button>'}
                    <button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothDisconnect('${device.address}')">Disconnect</button>
                </div>
            </div>
        `;
    }).join('');
}

function fetchConnectedDevices() {
    fetch('/bluetooth/connected')
        .then(r => r.json())
        .then(data => {
            bluetoothConnectedDevices = data.connected_devices || [];
            console.log('[BT] Connected devices fetched:', bluetoothConnectedDevices.length, bluetoothConnectedDevices);
            updateConnectedBluetoothUI();
        })
        .catch(e => console.error('Error fetching connected devices:', e));
}

function bluetoothConnect(address) {
    // Mark device as connecting
    bluetoothConnectingDevices.add(address);
    updateBluetoothUI();

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'bluetooth_connect',
            address: address
        }));
        // Removed hardcoded setTimeout - we now wait for bluetooth_connect_result via WS
    } else {
        bluetoothConnectingDevices.delete(address);
        updateBluetoothUI();
    }
}

function bluetoothSetActive(address) {
    fetch(`/bluetooth/set-active/${address}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            console.log('[BT] Set active response:', data);
            fetchConnectedDevices();
        })
        .catch(e => console.error('Error setting active device:', e));
}

function bluetoothDisconnect(address) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'bluetooth_disconnect',
            address: address
        }));
        // Fetch connected devices after a short delay
        setTimeout(fetchConnectedDevices, 500);
    }
}

function bluetoothRemove(address) {
    if (confirm(`Remove device ${address}?`)) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                action: 'bluetooth_remove',
                address: address
            }));
            // Fetch connected devices after removal
            setTimeout(fetchConnectedDevices, 500);
        }
    }
}

function startBluetoothScan() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        bluetoothScanning = true;
        bluetoothDevices = [];
        updateBluetoothUI();
        ws.send(JSON.stringify({
            action: 'bluetooth_start_scan'
        }));
    }
}

function stopBluetoothScan() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        bluetoothScanning = false;
        updateBluetoothUI();
        ws.send(JSON.stringify({
            action: 'bluetooth_stop_scan'
        }));
    }
}

// Bluetooth button event listeners
const btScanBtn = document.getElementById('btn_bt_scan');
if (btScanBtn) {
    btScanBtn.addEventListener('click', startBluetoothScan);
}

const btStopBtn = document.getElementById('btn_bt_stop');
if (btStopBtn) {
    btStopBtn.addEventListener('click', stopBluetoothScan);
}

// Bluetooth filter checkbox
const chkHideUnknown = document.getElementById('chk_hide_unknown');
if (chkHideUnknown) {
    chkHideUnknown.addEventListener('change', (e) => {
        hideUnknownDevices = e.target.checked;
        updateBluetoothUI();
    });
}

// Periodic polling of connected Bluetooth devices (more frequently initially)
// Fetch immediately on page load
fetchConnectedDevices();
setTimeout(fetchConnectedDevices, 300);
setTimeout(fetchConnectedDevices, 600);

// Then poll every 5 seconds (more relaxed for Orange Pi)
setInterval(fetchConnectedDevices, 5000);

connectWebSocket();
