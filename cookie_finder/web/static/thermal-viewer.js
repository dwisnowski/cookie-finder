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
const PAN_DISPLAY_MAX = 50;
const TILT_DISPLAY_MAX = 50;
const PAN_STEP = 5;
const TILT_STEP = 5;
const MOTOR_SPEED_MIN_HZ = 10;
const MOTOR_SPEED_MAX_HZ = 250;
const MOTOR_SPEED_DEFAULT_HZ = 100;
const VIDEO_ROTATION_STORAGE_KEY = 'cookieFinder.videoRotationDeg';
const PANTILT_ZERO_STORAGE_KEY = 'cookieFinder.panTiltZero';
const GAMEPAD_DEADZONE = 0.15;
const GAMEPAD_SENSITIVITY = 100;

let currentPan = 0;
let currentTilt = 0;
/** Absolute angles treated as relative 0,0 (graph center / HOME target). */
let homePan = 0;
let homeTilt = 0;
let motorActive = {};
let activeGamepadIndex = -1;
let connectedGamepads = [];
let lastGamepadPoll = Date.now();

let gamepadPanAxis = 0;
let gamepadTiltAxis = 1;
let gamepadInvertPan = false;
let gamepadInvertTilt = false;
/** Invert for on-screen / keyboard motor controls only (not gamepad). */
let uiInvertPan = false;
let uiInvertTilt = false;
const AXIS_NAMES = ['Left X', 'Left Y', 'Right X', 'Right Y'];
const UI_INVERT_STORAGE_KEY = 'cookieFinder.uiMotorInvert';

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

let wifiStatus = {
    supported: false,
    mode: 'unknown',
    ssid: null,
    ap_ssid: 'cookie-finder',
    ap_passphrase: null,
    open_network: true,
    ap_url: 'http://192.168.12.1/',
    switching: false,
    powering_off: false,
};
let wifiTargetMode = null;
let paletteAutoEnabled = false;

function isMobilePortrait() {
    return window.matchMedia('(max-width: 768px) and (orientation: portrait)').matches;
}

function maybeAutoEnablePalette() {
    if (paletteAutoEnabled || !isMobilePortrait()) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (state.palette_mode) {
        paletteAutoEnabled = true;
        return;
    }
    ws.send(JSON.stringify({ action: 'toggle_mode', mode: 'palette_mode' }));
    paletteAutoEnabled = true;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function sliderToHz(value) {
    const t = parseInt(value, 10) / 100;
    return Math.round(MOTOR_SPEED_MIN_HZ + t * (MOTOR_SPEED_MAX_HZ - MOTOR_SPEED_MIN_HZ));
}

function hzToSlider(hz) {
    const t = (hz - MOTOR_SPEED_MIN_HZ) / (MOTOR_SPEED_MAX_HZ - MOTOR_SPEED_MIN_HZ);
    return Math.round(clamp(t, 0, 1) * 100);
}

function getMotorSpeeds() {
    const panSlider = document.getElementById('slider_pan_speed')
        || document.getElementById('slider_pan_speed_modal');
    const tiltSlider = document.getElementById('slider_tilt_speed')
        || document.getElementById('slider_tilt_speed_modal');
    return {
        pan_hz: sliderToHz(panSlider ? panSlider.value : hzToSlider(MOTOR_SPEED_DEFAULT_HZ)),
        tilt_hz: sliderToHz(tiltSlider ? tiltSlider.value : hzToSlider(MOTOR_SPEED_DEFAULT_HZ)),
    };
}

function updateMotorSpeedLabels() {
    const speeds = getMotorSpeeds();
    document.querySelectorAll('.val-pan-speed-label').forEach(el => {
        el.textContent = speeds.pan_hz + ' Hz';
    });
    document.querySelectorAll('.val-tilt-speed-label').forEach(el => {
        el.textContent = speeds.tilt_hz + ' Hz';
    });
}

function syncMotorSpeedSliders(changedSlider, pairedId) {
    const paired = document.getElementById(pairedId);
    if (paired && paired.value !== changedSlider.value) {
        paired.value = changedSlider.value;
    }
}

function normalizeRotationDeg(deg) {
    return ((Math.round(Number(deg)) % 360) + 360) % 360;
}

function loadSavedVideoRotation() {
    try {
        const raw = localStorage.getItem(VIDEO_ROTATION_STORAGE_KEY);
        if (raw == null || raw === '') return 0;
        return normalizeRotationDeg(raw);
    } catch (_) {
        return 0;
    }
}

function saveVideoRotation(deg) {
    try {
        localStorage.setItem(VIDEO_ROTATION_STORAGE_KEY, String(normalizeRotationDeg(deg)));
    } catch (_) {
        /* ignore quota / private mode */
    }
}

function applyVideoRotation(deg) {
    const degrees = normalizeRotationDeg(deg);
    const img = document.getElementById('videoStream');
    const frame = document.getElementById('videoStreamFrame');
    const slider = document.getElementById('slider_video_rotation');
    const label = document.getElementById('val_video_rotation');

    if (slider && Number(slider.value) !== degrees) {
        slider.value = String(degrees);
    }
    if (label) {
        label.textContent = degrees + '°';
    }
    if (!img || !frame) return;

    // Layout size before transform (transform does not affect flow).
    const layoutW = img.clientWidth || frame.clientWidth;
    const naturalW = img.naturalWidth || layoutW;
    const naturalH = img.naturalHeight || layoutW;
    const layoutH = layoutW > 0 && naturalW > 0
        ? layoutW * (naturalH / naturalW)
        : img.clientHeight || layoutW;

    const rad = (degrees * Math.PI) / 180;
    const boundW = Math.abs(layoutW * Math.cos(rad)) + Math.abs(layoutH * Math.sin(rad));
    const boundH = Math.abs(layoutW * Math.sin(rad)) + Math.abs(layoutH * Math.cos(rad));
    const scale = boundW > 0 ? Math.min(1, frame.clientWidth / boundW) : 1;

    img.style.transform = `rotate(${degrees}deg) scale(${scale})`;
    frame.style.height = Math.max(1, boundH * scale) + 'px';
}

function setVideoRotation(deg, { persist = true } = {}) {
    const degrees = normalizeRotationDeg(deg);
    applyVideoRotation(degrees);
    if (persist) saveVideoRotation(degrees);
}

function nudgeVideoRotation(delta) {
    const slider = document.getElementById('slider_video_rotation');
    const current = slider ? Number(slider.value) : loadSavedVideoRotation();
    setVideoRotation(current + delta);
}

function sendMotorSpeed() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const speeds = getMotorSpeeds();
    ws.send(JSON.stringify({
        action: 'set_motor_speed',
        pan_hz: speeds.pan_hz,
        tilt_hz: speeds.tilt_hz,
    }));
}

function updateCameraSelector(data) {
    if (!data) return;
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
}

function loadPanTiltZero() {
    try {
        const raw = localStorage.getItem(PANTILT_ZERO_STORAGE_KEY);
        if (!raw) return { pan: 0, tilt: 0 };
        const parsed = JSON.parse(raw);
        return {
            pan: clamp(Number(parsed.pan) || 0, 0, PAN_MAX),
            tilt: clamp(Number(parsed.tilt) || 0, 0, TILT_MAX),
        };
    } catch (_) {
        return { pan: 0, tilt: 0 };
    }
}

function savePanTiltZero(pan, tilt) {
    try {
        localStorage.setItem(PANTILT_ZERO_STORAGE_KEY, JSON.stringify({ pan, tilt }));
    } catch (_) {
        /* ignore quota / private mode */
    }
}

function getRelativePanTilt() {
    return {
        pan: currentPan - homePan,
        tilt: currentTilt - homeTilt,
    };
}

function updatePanTiltIndicator() {
    const svgRadius = 75;
    const { pan: relPan, tilt: relTilt } = getRelativePanTilt();
    const panScale = PAN_DISPLAY_MAX || 1;
    const tiltScale = TILT_DISPLAY_MAX || 1;
    const x = 100 + clamp(relPan / panScale, -1, 1) * svgRadius;
    const y = 100 - clamp(relTilt / tiltScale, -1, 1) * svgRadius;

    const marker = document.getElementById('positionMarker');
    if (marker) {
        marker.setAttribute('cx', x);
        marker.setAttribute('cy', y);
        const lineH = document.getElementById('markerLineH');
        const lineV = document.getElementById('markerLineV');
        if (lineH) {
            lineH.setAttribute('x2', x);
            lineH.setAttribute('y2', y);
        }
        if (lineV) {
            lineV.setAttribute('x2', x);
            lineV.setAttribute('y2', y);
        }
    }

    document.querySelectorAll('.pan-angle-value').forEach(el => {
        el.textContent = relPan.toFixed(2) + '°';
    });
    document.querySelectorAll('.tilt-angle-value').forEach(el => {
        el.textContent = relTilt.toFixed(2) + '°';
    });

    const panPct = clamp((relPan + PAN_DISPLAY_MAX) / (2 * PAN_DISPLAY_MAX), 0, 1);
    const tiltPct = clamp((relTilt + TILT_DISPLAY_MAX) / (2 * TILT_DISPLAY_MAX), 0, 1);

    const panThumb = document.getElementById('panSliderThumb');
    if (panThumb) {
        panThumb.style.left = (panPct * 100) + '%';
    }
    const tiltThumb = document.getElementById('tiltSliderThumb');
    if (tiltThumb) {
        tiltThumb.style.bottom = (tiltPct * 100) + '%';
    }
}

function zeroPanTiltOrigin() {
    homePan = clamp(currentPan, 0, PAN_MAX);
    homeTilt = clamp(currentTilt, 0, TILT_MAX);
    savePanTiltZero(homePan, homeTilt);
    updatePanTiltIndicator();
}

function goPanTiltHome() {
    currentPan = homePan;
    currentTilt = homeTilt;
    updatePanTiltIndicator();

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'motor_command',
            command: 'motor_home',
            pan: homePan,
            tilt: homeTilt,
        }));
    }
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
        document.querySelectorAll(`.gamepad-button-display[data-gamepad-index="${i}"]`).forEach(buttonElement => {
            if (button && button.pressed) {
                buttonElement.classList.add('pressed');
            } else {
                buttonElement.classList.remove('pressed');
            }
        });
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
            currentPan = homePan;
            currentTilt = homeTilt;
            break;
    }

    updatePanTiltIndicator();
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(protocol + '://' + window.location.host + '/control');

    ws.onopen = () => {
        document.getElementById('statusText').innerHTML = 'Connected';
        sendMotorSpeed();
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
        } else if (msg.type === 'camera_status') {
            updateCameraStatus(msg.data);
        } else if (msg.type === 'available_cameras') {
            updateCameraSelector(msg.data);
        } else if (msg.type === 'bluetooth_connected') {
            applyBluetoothConnected(msg.data);
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
            } else if (btUpdate.status === 'scan_complete') {
                bluetoothScanning = false;
                bluetoothDevices = btUpdate.data.devices || [];
                updateBluetoothUI();
            } else if (btUpdate.status === 'device_connected' || btUpdate.status === 'device_disconnected' || btUpdate.status === 'device_removed') {
                if (btUpdate.data && btUpdate.data.devices) {
                    bluetoothDevices = btUpdate.data.devices;
                }
                updateBluetoothUI();
            }
        } else if (msg.type === 'bluetooth_state') {
            // Initial Bluetooth state on connection
            bluetoothDevices = msg.data.devices || [];
            bluetoothScanning = msg.data.scanning || false;
            updateBluetoothUI();
        } else if (msg.type === 'bluetooth_scan_started') {
            bluetoothScanning = true;
            updateBluetoothUI();
        } else if (msg.type === 'bluetooth_pair_result') {
            bluetoothPairingDevices.delete(msg.address);
            updateBluetoothUI();
            const statusDisplay = document.getElementById('btStatusDisplay');
            if (statusDisplay) {
                if (msg.success) {
                    statusDisplay.textContent = msg.message || `Paired ${msg.address}`;
                    statusDisplay.style.color = '';
                } else {
                    statusDisplay.textContent = msg.message || `Pair failed: ${msg.address}`;
                    statusDisplay.style.color = '#ff4444';
                    setTimeout(() => {
                        statusDisplay.style.color = '';
                        if (!bluetoothScanning) statusDisplay.textContent = 'Ready to scan';
                    }, 6000);
                }
            }
        } else if (msg.type === 'bluetooth_connect_result') {
            console.log('[BT] Connect result incoming:', msg);
            bluetoothConnectingDevices.delete(msg.address);
            updateBluetoothUI();
            const statusDisplay = document.getElementById('btStatusDisplay');
            if (!msg.success) {
                if (statusDisplay) {
                    statusDisplay.textContent = msg.message || `Connection failed to ${msg.address}`;
                    statusDisplay.style.color = '#ff4444';
                    setTimeout(() => {
                        statusDisplay.style.color = '';
                        if (!bluetoothScanning) statusDisplay.textContent = 'Ready to scan';
                    }, 6000);
                }
            } else if (statusDisplay) {
                statusDisplay.textContent = msg.message || 'Connected';
                statusDisplay.style.color = '';
            }
        } else if (msg.type === 'bluetooth_remove_result') {
            updateBluetoothUI();
            const statusDisplay = document.getElementById('btStatusDisplay');
            if (statusDisplay) {
                statusDisplay.textContent = msg.message || (msg.success ? 'Removed' : 'Remove failed');
                statusDisplay.style.color = msg.success ? '' : '#ff4444';
                setTimeout(() => {
                    statusDisplay.style.color = '';
                    if (!bluetoothScanning) statusDisplay.textContent = 'Ready to scan';
                }, 4000);
            }
        } else if (msg.type === 'bluetooth_scan_stopped') {
            bluetoothScanning = false;
            updateBluetoothUI();
        } else if (msg.type === 'bluetooth_disconnect_result') {
            updateBluetoothUI();
            if (!msg.success) {
                const statusDisplay = document.getElementById('btStatusDisplay');
                if (statusDisplay) {
                    statusDisplay.textContent = msg.message || 'Disconnect failed';
                    statusDisplay.style.color = '#ff4444';
                }
            }
        } else if (msg.type === 'wifi_status') {
            applyWifiStatus(msg.data);
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
    maybeAutoEnablePalette();
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

    updateWifiBadge();

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
            updateCameraSelector({
                available: availableCameras,
                current: newCameraId,
            });
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

const sliderPanSpeed = document.getElementById('slider_pan_speed');
const sliderTiltSpeed = document.getElementById('slider_tilt_speed');
const sliderPanSpeedModal = document.getElementById('slider_pan_speed_modal');
const sliderTiltSpeedModal = document.getElementById('slider_tilt_speed_modal');

function bindMotorSpeedSlider(slider, pairedId) {
    if (!slider) return;
    slider.addEventListener('input', () => {
        syncMotorSpeedSliders(slider, pairedId);
        updateMotorSpeedLabels();
        sendMotorSpeed();
    });
}

bindMotorSpeedSlider(sliderPanSpeed, 'slider_pan_speed_modal');
bindMotorSpeedSlider(sliderPanSpeedModal, 'slider_pan_speed');
bindMotorSpeedSlider(sliderTiltSpeed, 'slider_tilt_speed_modal');
bindMotorSpeedSlider(sliderTiltSpeedModal, 'slider_tilt_speed');
updateMotorSpeedLabels();

const sliderVideoRotation = document.getElementById('slider_video_rotation');
if (sliderVideoRotation) {
    sliderVideoRotation.addEventListener('input', (e) => {
        setVideoRotation(e.target.value);
    });
}
const btnRotateCcw = document.getElementById('btn_rotate_ccw');
if (btnRotateCcw) {
    btnRotateCcw.addEventListener('click', () => nudgeVideoRotation(-90));
}
const btnRotateCw = document.getElementById('btn_rotate_cw');
if (btnRotateCw) {
    btnRotateCw.addEventListener('click', () => nudgeVideoRotation(90));
}
const btnRotate180 = document.getElementById('btn_rotate_180');
if (btnRotate180) {
    btnRotate180.addEventListener('click', () => nudgeVideoRotation(180));
}
const btnRotateReset = document.getElementById('btn_rotate_reset');
if (btnRotateReset) {
    btnRotateReset.addEventListener('click', () => setVideoRotation(0));
}

const videoStreamEl = document.getElementById('videoStream');
if (videoStreamEl) {
    const refreshRotation = () => setVideoRotation(loadSavedVideoRotation(), { persist: false });
    videoStreamEl.addEventListener('load', refreshRotation);
    window.addEventListener('resize', refreshRotation);
    refreshRotation();
}

// Motor control
/** Maps logical UI command → resolved command currently running (handles invert mid-press). */
const uiMotorActiveResolved = {};

function resolveUiMotorCommand(command) {
    if ((command === 'motor_left' || command === 'motor_right') && uiInvertPan) {
        return command === 'motor_left' ? 'motor_right' : 'motor_left';
    }
    if ((command === 'motor_up' || command === 'motor_down') && uiInvertTilt) {
        return command === 'motor_up' ? 'motor_down' : 'motor_up';
    }
    return command;
}

function startMotorCommand(command) {
    const resolved = resolveUiMotorCommand(command);
    uiMotorActiveResolved[command] = resolved;
    motorActive[resolved] = true;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'motor_command',
            command: resolved,
            state: 'start'
        }));
    }
}

function stopMotorCommand(command) {
    const resolved = uiMotorActiveResolved[command] || resolveUiMotorCommand(command);
    delete uiMotorActiveResolved[command];
    motorActive[resolved] = false;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'motor_command',
            command: resolved,
            state: 'stop'
        }));
    }
}

function loadUiMotorInvert() {
    try {
        const raw = localStorage.getItem(UI_INVERT_STORAGE_KEY);
        if (!raw) return;
        const saved = JSON.parse(raw);
        uiInvertPan = !!saved.pan;
        uiInvertTilt = !!saved.tilt;
    } catch (_) {
        /* ignore corrupt storage */
    }
}

function saveUiMotorInvert() {
    try {
        localStorage.setItem(UI_INVERT_STORAGE_KEY, JSON.stringify({
            pan: uiInvertPan,
            tilt: uiInvertTilt
        }));
    } catch (_) {
        /* ignore quota / private mode */
    }
}

function syncUiMotorInvertToggles() {
    const panIds = ['toggle_invert_pan', 'toggle_invert_pan_modal'];
    const tiltIds = ['toggle_invert_tilt', 'toggle_invert_tilt_modal'];
    panIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = uiInvertPan;
    });
    tiltIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = uiInvertTilt;
    });
}

function bindUiMotorInvertToggles() {
    loadUiMotorInvert();
    syncUiMotorInvertToggles();

    const bindPair = (primaryId, modalId, setter) => {
        const primary = document.getElementById(primaryId);
        const modal = document.getElementById(modalId);
        const onChange = (source) => {
            setter(!!source.checked);
            syncUiMotorInvertToggles();
            saveUiMotorInvert();
        };
        if (primary) primary.addEventListener('change', () => onChange(primary));
        if (modal) modal.addEventListener('change', () => onChange(modal));
    };

    bindPair('toggle_invert_pan', 'toggle_invert_pan_modal', (v) => { uiInvertPan = v; });
    bindPair('toggle_invert_tilt', 'toggle_invert_tilt_modal', (v) => { uiInvertTilt = v; });
}

function bindMotorControl(el, command) {
    if (!el) return;

    const start = () => startMotorCommand(command);
    const stop = () => stopMotorCommand(command);

    el.addEventListener('mousedown', start);
    el.addEventListener('mouseup', stop);
    el.addEventListener('mouseleave', stop);
    el.addEventListener('touchstart', (e) => {
        e.preventDefault();
        start();
    });
    el.addEventListener('touchend', (e) => {
        e.preventDefault();
        stop();
    });
    el.addEventListener('touchcancel', (e) => {
        e.preventDefault();
        stop();
    });
}

const motorCommands = {
    'btn_motor_up': 'motor_up',
    'btn_motor_down': 'motor_down',
    'btn_motor_left': 'motor_left',
    'btn_motor_right': 'motor_right',
};

for (const [btnId, command] of Object.entries(motorCommands)) {
    bindMotorControl(document.getElementById(btnId), command);
}

document.querySelectorAll('.touch-zone[data-motor]').forEach(el => {
    bindMotorControl(el, el.dataset.motor);
});

document.querySelectorAll('.touch-zone-home[data-action="motor_home"]').forEach(el => {
    const goHome = (e) => {
        e.preventDefault();
        goPanTiltHome();
    };
    el.addEventListener('click', goHome);
});

function bindZeroHomeButtons(zeroId, homeId) {
    const zeroBtn = document.getElementById(zeroId);
    if (zeroBtn) {
        zeroBtn.addEventListener('click', (e) => {
            e.preventDefault();
            zeroPanTiltOrigin();
        });
    }
    const homeBtn = document.getElementById(homeId);
    if (homeBtn) {
        homeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            goPanTiltHome();
        });
    }
}

bindZeroHomeButtons('btn_motor_zero', 'btn_motor_home');
bindZeroHomeButtons('btn_motor_zero_modal', 'btn_motor_home_modal');
bindUiMotorInvertToggles();

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
            startMotorCommand(motorMap[key]);
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

        keyPressState[key] = false;
        stopMotorCommand(motorMap[key]);
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

function updateCameraStatus(data) {
    if (!data) return;
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
}

// Settings modal
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsBtn = document.getElementById('btn_settings');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');

function applyWifiStatus(data) {
    wifiStatus = Object.assign({}, wifiStatus, data || {});
    updateWifiBadge();
    updateWifiSettingsUI();
}

function updateWifiBadge() {
    const badge = document.getElementById('badge_wifi');
    const dot = document.getElementById('badge_wifi_dot');
    const text = document.getElementById('badge_wifi_text');
    if (!badge || !dot || !text) return;

    badge.classList.remove('badge-wifi-ap', 'badge-wifi-client', 'badge-wifi-unknown');
    if (wifiStatus.powering_off) {
        badge.classList.add('badge-wifi-unknown');
        dot.classList.remove('active');
        text.textContent = 'Shutting down…';
        return;
    }
    if (wifiStatus.switching) {
        badge.classList.add('badge-wifi-unknown');
        dot.classList.remove('active');
        text.textContent = 'WiFi switching…';
        return;
    }

    if (wifiStatus.mode === 'ap') {
        badge.classList.add('badge-wifi-ap');
        dot.classList.add('active');
        text.textContent = `AP · ${wifiStatus.ap_ssid || 'cookie-finder'}`;
        return;
    }

    if (wifiStatus.mode === 'client') {
        badge.classList.add('badge-wifi-client');
        dot.classList.add('active');
        const ssid = wifiStatus.ssid ? wifiStatus.ssid : 'network';
        text.textContent = `Client · ${ssid}`;
        return;
    }

    badge.classList.add('badge-wifi-unknown');
    dot.classList.remove('active');
    text.textContent = wifiStatus.supported === false ? 'WiFi n/a' : 'WiFi —';
}

function updateWifiSettingsUI() {
    const summary = document.getElementById('wifiModeSummary');
    const toggleBtn = document.getElementById('btn_wifi_toggle');
    const hint = document.getElementById('wifiModeHint');
    if (!summary || !toggleBtn) return;

    if (wifiStatus.supported === false) {
        summary.textContent = wifiStatus.reason || 'WiFi AP mode is not available on this device.';
        toggleBtn.disabled = true;
        toggleBtn.textContent = 'AP Mode Unavailable';
        if (hint) hint.textContent = 'Requires Orange Pi Linux with sudo access for scripts/wifi-mode.sh.';
        return;
    }

    if (wifiStatus.powering_off) {
        summary.textContent = 'The Orange Pi is shutting down. Watch the WiFi LED (slow → fast → slow).';
        toggleBtn.disabled = true;
        toggleBtn.textContent = 'Shutting down…';
        const shutdownBtn = document.getElementById('btn_poweroff_settings');
        if (shutdownBtn) shutdownBtn.disabled = true;
        return;
    }

    if (wifiStatus.switching) {
        const pending = wifiStatus.pending_mode || 'new';
        summary.textContent = `Switching to ${pending} mode… You may lose this connection shortly.`;
        toggleBtn.disabled = true;
        toggleBtn.textContent = 'Switching…';
        return;
    }

    if (wifiStatus.mode === 'ap') {
        summary.textContent = `Currently hosting access point “${wifiStatus.ap_ssid || 'cookie-finder'}” (${wifiStatus.ap_gateway || '192.168.12.1'}).`;
        toggleBtn.disabled = false;
        toggleBtn.textContent = 'Switch to Client Mode';
        if (hint) {
            hint.innerHTML = `Open network (no password) · <strong>${wifiStatus.ap_url || 'http://192.168.12.1/'}</strong>`;
        }
        return;
    }

    if (wifiStatus.mode === 'client') {
        const ssid = wifiStatus.ssid ? `“${wifiStatus.ssid}”` : 'a saved WiFi network';
        summary.textContent = `Currently connected as a WiFi client to ${ssid}.`;
        toggleBtn.disabled = false;
        toggleBtn.textContent = 'Switch to AP Mode (cookie-finder)';
        if (hint) {
            hint.innerHTML = `AP network name: <strong>${wifiStatus.ap_ssid || 'cookie-finder'}</strong>`;
        }
        return;
    }

    summary.textContent = 'WiFi mode could not be determined.';
    toggleBtn.disabled = false;
    toggleBtn.textContent = 'Try Switch to AP Mode';
}

function refreshWifiStatus() {
    fetch('/wifi/status')
        .then((r) => r.json())
        .then((data) => applyWifiStatus(data))
        .catch((e) => console.error('WiFi status error:', e));
}

function openWifiConfirm(targetMode, instructions) {
    wifiTargetMode = targetMode;
    const overlay = document.getElementById('wifiConfirmOverlay');
    const title = document.getElementById('wifiConfirmTitle');
    const summary = document.getElementById('wifiConfirmSummary');
    const steps = document.getElementById('wifiConfirmSteps');
    const creds = document.getElementById('wifiConfirmCreds');
    const confirmBtn = document.getElementById('btn_wifi_confirm');
    if (!overlay || !title || !summary || !steps || !confirmBtn) return;

    title.textContent = instructions.title || 'Switch WiFi Mode?';
    summary.textContent = instructions.summary || '';
    steps.innerHTML = '';
    (instructions.steps || []).forEach((step) => {
        const li = document.createElement('li');
        li.textContent = step;
        steps.appendChild(li);
    });

    if (creds) {
        if (targetMode === 'ap') {
            creds.hidden = false;
            const passRow = instructions.open_network || !instructions.passphrase
                ? `<div><span>Security</span><strong>Open (no password)</strong></div>`
                : `<div><span>Password</span><strong>${instructions.passphrase}</strong></div>`;
            creds.innerHTML = `
                <div><span>Network</span><strong>${instructions.ssid || 'cookie-finder'}</strong></div>
                ${passRow}
                <div><span>URL</span><strong>${instructions.url || 'http://192.168.12.1/'}</strong></div>
            `;
        } else {
            creds.hidden = true;
            creds.innerHTML = '';
        }
    }

    confirmBtn.disabled = false;
    confirmBtn.textContent = targetMode === 'ap' ? 'Switch to AP Mode' : 'Switch to Client Mode';
    overlay.classList.add('active');
}

function closeWifiConfirm() {
    const overlay = document.getElementById('wifiConfirmOverlay');
    if (overlay) overlay.classList.remove('active');
    wifiTargetMode = null;
}

function requestWifiModeSwitch() {
    if (!wifiStatus.supported) return;
    const target = wifiStatus.mode === 'ap' ? 'client' : 'ap';
    fetch(`/wifi/instructions/${target}`)
        .then((r) => r.json())
        .then((instructions) => {
            if (instructions.error) {
                alert(instructions.error);
                return;
            }
            openWifiConfirm(target, instructions);
        })
        .catch((e) => {
            console.error('WiFi instructions error:', e);
            alert('Could not load WiFi switch instructions.');
        });
}

function confirmWifiModeSwitch() {
    if (!wifiTargetMode) return;
    const confirmBtn = document.getElementById('btn_wifi_confirm');
    const cancelBtn = document.getElementById('btn_wifi_cancel');
    if (confirmBtn) {
        confirmBtn.disabled = true;
        confirmBtn.textContent = 'Switching…';
    }
    if (cancelBtn) cancelBtn.disabled = true;

    fetch(`/wifi/mode/${wifiTargetMode}`, { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
            if (data.wifi) applyWifiStatus(data.wifi);
            if (data.status === 'error' || data.status === 'busy') {
                alert(data.message || 'WiFi switch failed');
                if (confirmBtn) {
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = 'Switch Mode';
                }
                if (cancelBtn) cancelBtn.disabled = false;
                return;
            }

            const summary = document.getElementById('wifiConfirmSummary');
            if (summary) {
                summary.textContent = (data.message || 'Switch started.') +
                    ' Keep these instructions handy — this page will disconnect when the radio changes.';
            }
            if (confirmBtn) confirmBtn.textContent = 'Switch started';
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.textContent = 'Close';
            }
        })
        .catch((e) => {
            console.error('WiFi switch error:', e);
            // Likely disconnected mid-switch; leave instructions visible
            const summary = document.getElementById('wifiConfirmSummary');
            if (summary) {
                summary.textContent = 'Connection lost while switching. Follow the steps above to reconnect.';
            }
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.textContent = 'Close';
            }
        });
}

function openSettings() {
    settingsOverlay.classList.add('active');
    refreshWifiStatus();
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

// Connect / QR modal
const connectOverlay = document.getElementById('connectOverlay');
const connectBtn = document.getElementById('btn_connect');
const closeConnectBtn = document.getElementById('closeConnectBtn');

function renderConnectQr(url) {
    const holder = document.getElementById('connectQr');
    if (!holder) return;
    holder.innerHTML = '';
    if (!url || typeof qrcode !== 'function') {
        holder.textContent = 'QR unavailable';
        return;
    }
    try {
        const qr = qrcode(0, 'M');
        qr.addData(url);
        qr.make();
        holder.innerHTML = qr.createSvgTag(4, 0);
        holder.setAttribute('title', url);
    } catch (err) {
        console.error('QR render error:', err);
        holder.textContent = 'QR unavailable';
    }
}

function applyConnectInfo(data) {
    const ipEl = document.getElementById('connectIp');
    const mdnsEl = document.getElementById('connectMdns');
    const hintEl = document.getElementById('connectHint');
    const url = (data && data.url) || window.location.origin + '/';
    const ip = data && data.ip;
    const mdnsUrl = (data && data.mdns_url) || 'http://cookie-finder.local/';

    if (ipEl) {
        if (ip) {
            const ipUrl = `http://${ip}/`;
            ipEl.textContent = ipUrl;
            if (ipEl.tagName === 'A') ipEl.href = ipUrl;
        } else {
            ipEl.textContent = 'Not available';
            if (ipEl.tagName === 'A') ipEl.removeAttribute('href');
        }
    }
    if (mdnsEl) {
        mdnsEl.textContent = mdnsUrl;
        if (mdnsEl.tagName === 'A') mdnsEl.href = mdnsUrl;
    }
    if (hintEl) {
        if (data && data.wifi_mode === 'ap') {
            hintEl.textContent = 'Access-point mode — connect to the cookie-finder WiFi, then scan.';
        } else if (data && data.addresses && data.addresses.length > 1) {
            const extras = data.addresses
                .filter((a) => a.ip !== ip)
                .map((a) => `${a.interface}: ${a.ip}`)
                .join(' · ');
            hintEl.textContent = extras ? `Also: ${extras}` : '';
        } else {
            hintEl.textContent = '';
        }
    }
    renderConnectQr(url);
}

function openConnect() {
    if (!connectOverlay) return;
    connectOverlay.classList.add('active');
    const ipEl = document.getElementById('connectIp');
    if (ipEl) ipEl.textContent = 'Loading…';
    fetch('/network/info')
        .then((r) => r.json())
        .then((data) => applyConnectInfo(data))
        .catch((e) => {
            console.error('Network info error:', e);
            applyConnectInfo({
                url: window.location.origin + '/',
                ip: null,
                mdns_url: 'http://cookie-finder.local/',
            });
        });
}

function closeConnect() {
    if (connectOverlay) connectOverlay.classList.remove('active');
}

if (connectBtn) connectBtn.addEventListener('click', openConnect);
if (closeConnectBtn) closeConnectBtn.addEventListener('click', closeConnect);
if (connectOverlay) {
    connectOverlay.addEventListener('click', (e) => {
        if (e.target === connectOverlay) closeConnect();
    });
}

// Motor control modal
const motorOverlay = document.getElementById('motorOverlay');
const motorBtn = document.getElementById('btn_motor');
const closeMotorBtn = document.getElementById('closeMotorBtn');

function openMotor() {
    if (motorOverlay) motorOverlay.classList.add('active');
}

function closeMotor() {
    if (motorOverlay) motorOverlay.classList.remove('active');
}

if (motorBtn) motorBtn.addEventListener('click', openMotor);
if (closeMotorBtn) closeMotorBtn.addEventListener('click', closeMotor);
if (motorOverlay) {
    motorOverlay.addEventListener('click', (e) => {
        if (e.target === motorOverlay) {
            closeMotor();
        }
    });
}

const wifiToggleBtn = document.getElementById('btn_wifi_toggle');
if (wifiToggleBtn) {
    wifiToggleBtn.addEventListener('click', requestWifiModeSwitch);
}

const wifiConfirmOverlay = document.getElementById('wifiConfirmOverlay');
const closeWifiConfirmBtn = document.getElementById('closeWifiConfirmBtn');
const wifiCancelBtn = document.getElementById('btn_wifi_cancel');
const wifiConfirmBtn = document.getElementById('btn_wifi_confirm');
if (closeWifiConfirmBtn) closeWifiConfirmBtn.addEventListener('click', closeWifiConfirm);
if (wifiCancelBtn) wifiCancelBtn.addEventListener('click', closeWifiConfirm);
if (wifiConfirmBtn) wifiConfirmBtn.addEventListener('click', confirmWifiModeSwitch);
if (wifiConfirmOverlay) {
    wifiConfirmOverlay.addEventListener('click', (e) => {
        if (e.target === wifiConfirmOverlay) closeWifiConfirm();
    });
}

refreshWifiStatus();
setInterval(refreshWifiStatus, 5000);

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

({ pan: homePan, tilt: homeTilt } = loadPanTiltZero());
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
            .catch(e => console.error('Reconnect error:', e));
    });
}

function openPoweroffConfirm() {
    const overlay = document.getElementById('poweroffConfirmOverlay');
    const summary = document.getElementById('poweroffConfirmSummary');
    const confirmBtn = document.getElementById('btn_poweroff_confirm');
    const cancelBtn = document.getElementById('btn_poweroff_cancel');
    if (!overlay) return;
    if (summary) {
        summary.textContent =
            'The WiFi LED will pulse slow → fast → slow, then the Orange Pi will power off. ' +
            'To turn it back on, unplug and replug power.';
    }
    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = 'Shut down';
    }
    if (cancelBtn) {
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Cancel';
    }
    overlay.classList.add('active');
}

function closePoweroffConfirm() {
    const overlay = document.getElementById('poweroffConfirmOverlay');
    if (overlay) overlay.classList.remove('active');
}

function confirmPoweroff() {
    const confirmBtn = document.getElementById('btn_poweroff_confirm');
    const cancelBtn = document.getElementById('btn_poweroff_cancel');
    const closeBtn = document.getElementById('closePoweroffConfirmBtn');
    const settingsShutdownBtn = document.getElementById('btn_poweroff_settings');
    if (confirmBtn) {
        confirmBtn.disabled = true;
        confirmBtn.textContent = 'Shutting down…';
    }
    if (cancelBtn) cancelBtn.disabled = true;
    if (closeBtn) closeBtn.disabled = true;
    if (settingsShutdownBtn) settingsShutdownBtn.disabled = true;

    fetch('/system/poweroff', { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
            if (data.powering_off) {
                applyWifiStatus({ powering_off: true });
            }
            if (data.status === 'error') {
                alert(data.message || 'Shutdown failed');
                if (confirmBtn) {
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = 'Shut down';
                }
                if (cancelBtn) cancelBtn.disabled = false;
                if (closeBtn) closeBtn.disabled = false;
                if (settingsShutdownBtn) settingsShutdownBtn.disabled = false;
                return;
            }
            const summary = document.getElementById('poweroffConfirmSummary');
            if (summary) {
                summary.textContent = data.message ||
                    'Shutting down… watch the WiFi LED (slow → fast → slow). This page will disconnect.';
            }
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.textContent = 'Close';
            }
        })
        .catch((e) => {
            console.error('Poweroff error:', e);
            const summary = document.getElementById('poweroffConfirmSummary');
            if (summary) {
                summary.textContent =
                    'Connection lost — the Orange Pi is likely powering off. Watch the WiFi LED.';
            }
            applyWifiStatus({ powering_off: true });
        });
}

const poweroffSettingsBtn = document.getElementById('btn_poweroff_settings');
if (poweroffSettingsBtn) {
    poweroffSettingsBtn.addEventListener('click', openPoweroffConfirm);
}
const poweroffConfirmOverlay = document.getElementById('poweroffConfirmOverlay');
const closePoweroffConfirmBtn = document.getElementById('closePoweroffConfirmBtn');
const poweroffCancelBtn = document.getElementById('btn_poweroff_cancel');
const poweroffConfirmBtn = document.getElementById('btn_poweroff_confirm');
if (closePoweroffConfirmBtn) closePoweroffConfirmBtn.addEventListener('click', closePoweroffConfirm);
if (poweroffCancelBtn) poweroffCancelBtn.addEventListener('click', closePoweroffConfirm);
if (poweroffConfirmBtn) poweroffConfirmBtn.addEventListener('click', confirmPoweroff);
if (poweroffConfirmOverlay) {
    poweroffConfirmOverlay.addEventListener('click', (e) => {
        if (e.target === poweroffConfirmOverlay) closePoweroffConfirm();
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

// === PI BLUETOOTH GAMEPAD (BlueZ HID on the robot) ===
let bluetoothDevices = [];
let bluetoothScanning = false;
let hideUnknownDevices = true; // Default: hide Unknown Device
let bluetoothConnectedDevices = [];
let bluetoothConnectingDevices = new Set(); // Track devices being connected
let bluetoothPairingDevices = new Set();

function bluetoothDeviceStatusText(device) {
    if (device.connected) return 'Connected';
    if (device.paired) return 'Paired';
    return 'Available';
}

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
        const statusText = bluetoothDeviceStatusText(device);
        const isConnecting = bluetoothConnectingDevices.has(device.address);
        const isPairing = bluetoothPairingDevices.has(device.address);
        const busy = isConnecting || isPairing;
        let actionButtons = '';
        if (!device.paired) {
            actionButtons += `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px; ${busy ? 'opacity: 0.6; cursor: not-allowed;' : ''}" onclick="bluetoothPair('${device.address}')" ${busy ? 'disabled' : ''}>${isPairing ? 'Pairing...' : 'Pair'}</button>`;
        }
        if (!device.connected) {
            actionButtons += `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px; ${busy ? 'opacity: 0.6; cursor: not-allowed;' : ''}" onclick="bluetoothConnect('${device.address}')" ${busy ? 'disabled' : ''}>${isConnecting ? 'Connecting...' : 'Connect'}</button>`;
        } else {
            actionButtons += `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothDisconnect('${device.address}')">Disconnect</button>`;
        }
        actionButtons += `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothSetActive('${device.address}')">Set Active</button>`;
        actionButtons += `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothRemove('${device.address}')">Remove</button>`;

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
                        ${addressShort}
                    </div>
                </div>
                <div style="margin-bottom: 4px; font-size: 10px; color: var(--accent);">
                    ${isConnecting ? 'Connecting...' : isPairing ? 'Pairing...' : statusText}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3px;">
                    ${actionButtons}
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
                        ${device.name || 'Unknown Device'}
                        ${isActive ? '(active)' : ''}
                    </div>
                    <div style="font-size: 9px; color: var(--text-tertiary);">
                        ${addressShort}
                    </div>
                </div>
                ${isActive ? '<div style="font-size: 9px; color: var(--accent); margin-bottom: 4px; font-weight: bold;">ACTIVE INPUT</div>' : ''}
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3px;">
                    ${!isActive ? `<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothSetActive('${device.address}')">Set Active</button>` : '<button class="btn-toggle" style="padding: 4px 6px; font-size: 9px; opacity: 0.5; cursor: default;">Active</button>'}
                    <button class="btn-toggle" style="padding: 4px 6px; font-size: 9px;" onclick="bluetoothDisconnect('${device.address}')">Disconnect</button>
                </div>
            </div>
        `;
    }).join('');
}

function applyBluetoothConnected(data) {
    if (!data) return;
    bluetoothConnectedDevices = data.connected_devices || [];
    updateConnectedBluetoothUI();
}

function bluetoothPair(address) {
    bluetoothPairingDevices.add(address);
    updateBluetoothUI();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'bluetooth_pair',
            address: address
        }));
    } else {
        bluetoothPairingDevices.delete(address);
        updateBluetoothUI();
    }
}

function bluetoothConnect(address) {
    bluetoothConnectingDevices.add(address);
    updateBluetoothUI();

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'bluetooth_connect',
            address: address
        }));
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
            const statusDisplay = document.getElementById('btStatusDisplay');
            if (statusDisplay && data.message) {
                statusDisplay.textContent = data.message;
                statusDisplay.style.color = data.status === 'success' ? '' : '#ff4444';
            }
        })
        .catch(e => console.error('Error setting active device:', e));
}

function bluetoothDisconnect(address) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: 'bluetooth_disconnect',
            address: address
        }));
    }
}

function bluetoothRemove(address) {
    if (confirm(`Remove (unpair) device ${address} from this robot?`)) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                action: 'bluetooth_remove',
                address: address
            }));
        }
    }
}

function startBluetoothScan() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        bluetoothScanning = true;
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

connectWebSocket();
