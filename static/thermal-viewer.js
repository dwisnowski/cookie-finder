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

function updateCameraSelector() {
    fetch('/available-cameras')
        .then(r => r.json())
        .then(data => {
            availableCameras = data.available;
            currentCamera = data.current;
            
            const selector = document.getElementById('cameraSelector');
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
            
            document.getElementById('currentCameraId').textContent = currentCamera !== null ? currentCamera : '--';
        })
        .catch(e => console.error('Failed to fetch cameras:', e));
}

function updatePanTiltIndicator() {
    const svgRadius = 75;
    const panPercent = currentPan / PAN_MAX;
    const tiltPercent = currentTilt / TILT_MAX;
    
    const x = 100 + (panPercent * svgRadius);
    const y = 100 - (tiltPercent * svgRadius);
    
    const marker = document.getElementById('positionMarker');
    marker.setAttribute('cx', x);
    marker.setAttribute('cy', y);
    
    document.getElementById('markerLineH').setAttribute('x2', x);
    document.getElementById('markerLineH').setAttribute('y2', y);
    document.getElementById('markerLineV').setAttribute('x2', x);
    document.getElementById('markerLineV').setAttribute('y2', y);
    
    document.getElementById('panAngle').textContent = currentPan + '°';
    document.getElementById('tiltAngle').textContent = currentTilt + '°';
}

function updateGamepadStatus() {
    const gamepads = navigator.getGamepads?.() || [];
    connectedGamepads = Array.from(gamepads).filter(gp => gp !== null);
    
    const indicator = document.getElementById('gamepadIndicator');
    const statusText = document.getElementById('gamepadStatusText');
    const nameDisplay = document.getElementById('gamepadNameDisplay');
    const countDisplay = document.getElementById('gamepadDeviceCount');
    
    countDisplay.textContent = connectedGamepads.length + ' device' + (connectedGamepads.length !== 1 ? 's' : '');
    
    if (activeGamepadIndex >= 0 && activeGamepadIndex < connectedGamepads.length) {
        const activeGpad = connectedGamepads[activeGamepadIndex];
        indicator.classList.add('connected');
        statusText.textContent = '✓ Connected';
        nameDisplay.textContent = activeGpad.id;
    } else if (connectedGamepads.length > 0) {
        activeGamepadIndex = 0;
        const activeGpad = connectedGamepads[0];
        indicator.classList.add('connected');
        statusText.textContent = '✓ Connected';
        nameDisplay.textContent = activeGpad.id;
    } else {
        activeGamepadIndex = -1;
        indicator.classList.remove('connected');
        statusText.textContent = '✗ Disconnected';
        nameDisplay.textContent = '—';
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
        
        currentPan = Math.max(-PAN_MAX, Math.min(PAN_MAX, currentPan + panChange));
        currentTilt = Math.max(-TILT_MAX, Math.min(TILT_MAX, currentTilt + tiltChange));
        
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
    
    switch(command) {
        case 'motor_left':
            currentPan = Math.max(-PAN_MAX, currentPan - increment);
            break;
        case 'motor_right':
            currentPan = Math.min(PAN_MAX, currentPan + increment);
            break;
        case 'motor_up':
            currentTilt = Math.min(TILT_MAX, currentTilt + increment);
            break;
        case 'motor_down':
            currentTilt = Math.max(-TILT_MAX, currentTilt - increment);
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
    };
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
            state = msg.data;
            updateUI();
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

let motorIntervals = {};

for (const [btnId, command] of Object.entries(motorCommands)) {
    const btn = document.getElementById(btnId);
    if (btn) {
        const startMotor = () => {
            motorActive[command] = true;
            updateMotorAngle(command);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    action: 'motor_command',
                    command: command,
                    state: 'start'
                }));
            }
            
            motorIntervals[command] = setInterval(() => {
                if (motorActive[command]) {
                    updateMotorAngle(command);
                }
            }, 100);
        };
        
        const stopMotor = () => {
            motorActive[command] = false;
            if (motorIntervals[command]) {
                clearInterval(motorIntervals[command]);
                delete motorIntervals[command];
            }
            
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
            updateMotorAngle(command);
            
            ws.send(JSON.stringify({
                action: 'motor_command',
                command: command,
                state: 'start'
            }));
            
            motorIntervals[command] = setInterval(() => {
                if (motorActive[command]) {
                    updateMotorAngle(command);
                }
            }, 100);
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
        if (motorIntervals[command]) {
            clearInterval(motorIntervals[command]);
            delete motorIntervals[command];
        }
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
            
            if (data.connected) {
                indicator.style.background = '#00aa00';
                message.textContent = '✓ Camera Connected';
            } else {
                indicator.style.background = '#ff4444';
                message.textContent = '✗ Camera Disconnected';
            }
        })
        .catch(e => console.error('Status fetch error:', e));
}

// Reconnect button
document.getElementById('btn_reconnect').addEventListener('click', () => {
    fetch('/reconnect', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            for (let i = 0; i < 10; i++) {
                setTimeout(updateCameraStatus, i * 500);
            }
        })
        .catch(e => console.error('Reconnect error:', e));
});

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

const cycleBtn = document.getElementById('btn_cycle_gamepad');
if (cycleBtn) {
    cycleBtn.addEventListener('click', cycleGamepad);
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

connectWebSocket();
