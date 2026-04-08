"""
Bluetooth device scanning and connection management using bleak.
Async-friendly BLE (Bluetooth Low Energy) controller for Orange Pi and other platforms.
"""

import asyncio
import threading
import traceback
import subprocess
from typing import List, Dict, Optional, Callable, Set
from bleak import BleakScanner, BleakClient, BleakError

try:
    import pygame
except ImportError:
    pygame = None  # Fallback if pygame not installed


class BluetoothDevice:
    """Represents a discovered Bluetooth device."""
    
    def __init__(self, address: str, name: str, rssi: int, advertised_services: List[str] = None):
        self.address = address
        self.name = name or f"Unknown ({address[-5:]})"
        self.rssi = rssi  # Signal strength in dBm
        self.paired = False  # bleak doesn't track pairing state directly
        self.connected = False
        self.advertised_services = advertised_services or []
    
    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "name": self.name,
            "rssi": self.rssi,
            "paired": self.paired,
            "connected": self.connected,
            "signal_strength": self._rssi_to_bars(self.rssi),
            "advertised_services": len(self.advertised_services)
        }
    
    @staticmethod
    def _rssi_to_bars(rssi: int) -> str:
        """Convert RSSI to signal strength bars."""
        if rssi >= -50:
            return "▓▓▓▓▓"
        elif rssi >= -60:
            return "▓▓▓▓░"
        elif rssi >= -70:
            return "▓▓▓░░"
        elif rssi >= -80:
            return "▓▓░░░"
        else:
            return "▓░░░░"


class BluetoothController:
    """Manages Bluetooth device discovery and connection using bleak."""
    
    def __init__(self):
        self.scanning = False
        self.devices: Dict[str, BluetoothDevice] = {}
        self.connected_devices: Dict[str, BleakClient] = {}
        self.connected_device_address: Optional[str] = None  # Track active input device
        self.scan_thread: Optional[threading.Thread] = None
        self.status_callback: Optional[Callable] = None
        self.last_input_data: Dict = {"pan_axis": 0.0, "tilt_axis": 0.0}  # Cached input
        self.notification_data: Dict[str, bytes] = {}  # Store latest notification data per characteristic
        self.joystick_thread: Optional[threading.Thread] = None  # Thread for reading joystick
        self.joystick_running = False
        self.last_joystick_input: Dict = {"pan_axis": 0.0, "tilt_axis": 0.0, "buttons": {}}  # Latest joystick state
    
    def _get_system_connected_devices(self) -> Set[str]:
        """Query system Bluetooth adapter for actually connected devices (Linux)."""
        connected = set()
        try:
            # Try to get connected devices from bluetoothctl
            result = subprocess.run(
                ['bluetoothctl', '--', 'devices', 'Connected'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output: "Device <address> <name>"
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Device '):
                        parts = line.split()
                        if len(parts) >= 2:
                            address = parts[1]
                            connected.add(address.upper())
                print(f"[BT] System-connected devices: {connected}")
            else:
                print(f"[BT] bluetoothctl returned code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"[BT] Error querying system devices: {e}")
        
        return connected
    
    def _get_device_name_from_system(self, address: str) -> Optional[str]:
        """Query device name from bluetoothctl."""
        try:
            # Use bluetoothctl to get device info
            result = subprocess.run(
                ['bluetoothctl', '--', 'info', address],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output looking for "Name: <device name>"
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Name:'):
                        # Extract name after "Name: "
                        name = line.replace('Name:', '').strip()
                        if name:
                            return name
        except Exception:
            pass
        
        return None
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates."""
        self.status_callback = callback
    
    def _emit_status(self, status: str, data: Optional[Dict] = None):
        """Emit status update via callback."""
        if self.status_callback:
            self.status_callback({
                "status": status,
                "data": data or {}
            })
    
    def _run_async(self, coro, timeout=12):
        """Helper to run async code from sync context."""
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread
                loop = None
            
            if loop and loop.is_running():
                # Loop is already running, schedule as task
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=timeout)
            elif loop:
                # Loop exists but not running, use it
                return loop.run_until_complete(coro)
            else:
                # No loop, create new one
                return asyncio.run(coro)
        except RuntimeError as e:
            if "different loop" in str(e):
                # Event loop conflict - just log and continue
                print(f"[BT] Event loop warning (non-critical): {e}")
                return False
            raise
    
    def start_scan(self) -> bool:
        """Start scanning for Bluetooth devices."""
        if self.scanning:
            return False
        
        self.scanning = True
        self.devices = {}
        
        print("[BT] Starting device scan...")
        self._emit_status("scan_started")
        
        self.scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self.scan_thread.start()
        
        return True
    
    def stop_scan(self):
        """Stop Bluetooth scanning."""
        self.scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=3)
        
        print("[BT] Scan stopped")
        self._emit_status("scan_stopped", {"devices": self.get_devices_list()})
    
    def _scan_worker(self):
        """Background worker for Bluetooth scanning."""
        try:
            self._run_async(self._scan_async())
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[BT] Scan error: {error_msg}")
            print(f"[BT] Traceback:\n{tb}")
            self.scanning = False
            self._emit_status("scan_error", {"error": error_msg})
    
    async def _scan_async(self):
        """Async scanning using bleak."""
        try:
            print("[BT] Scanning for BLE devices...")
            
            # Scan for devices - detection_callback is called as devices are found
            def on_detection(device, advertisement_data):
                """Callback when a device is discovered."""
                address = device.address
                
                # Extract name from advertisement data (multiple possible locations)
                name = None
                if advertisement_data.local_name:
                    name = advertisement_data.local_name
                elif device.name:
                    name = device.name
                
                # Fallback to address if no name found
                if not name:
                    name = f"Unknown Device ({address[-5:]})"
                
                rssi = advertisement_data.rssi or -100
                services = list(advertisement_data.service_uuids) if advertisement_data.service_uuids else []
                
                ble_device = BluetoothDevice(address, name, rssi, services)
                self.devices[address] = ble_device
                
                # Emit periodic update
                devices_list = self.get_devices_list()
                self._emit_status("scan_update", {"devices": devices_list})
                print(f"[BT] Found: {name} ({address}) RSSI={rssi}")
            
            # Run scanner for ~6 seconds
            async with BleakScanner(detection_callback=on_detection) as scanner:
                await asyncio.sleep(6)
            
            print(f"[BT] Scan complete. Found {len(self.devices)} devices")
            self.scanning = False
            self._emit_status("scan_complete", {"devices": self.get_devices_list()})
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[BT] Scan error: {error_msg}")
            print(f"[BT] Traceback:\n{tb}")
            self.scanning = False
            self._emit_status("scan_error", {"error": error_msg})
    
    def connect_device(self, address: str, retries: int = 1) -> bool:
        """Attempt to connect to a device with optional retries."""
        for attempt in range(1, retries + 1):
            try:
                if attempt > 1:
                    print(f"[BT] Retry attempt {attempt}/{retries} for {address}")
                print(f"[BT] Connecting to {address}...")
                result = self._run_async(self._connect_async(address))
                if result:
                    return True
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                tb = traceback.format_exc()
                print(f"[BT] Connection error (attempt {attempt}/{retries}): {error_msg}")
                print(f"[BT] Traceback:\n{tb}")
                
                if attempt < retries:
                    print(f"[BT] Waiting 2 seconds before retry...")
                    import time
                    time.sleep(2)
                else:
                    self._emit_status("device_connect_error", {"address": address, "error": error_msg})
        
        return False
    
    async def _connect_async(self, address: str) -> bool:
        """Async connect to device."""
        try:
            # Check if already connected
            if address in self.connected_devices:
                print(f"[BT] Already connected to {address}")
                self.connected_device_address = address
                return True
            
            # Create and connect client
            device = self.devices.get(address)
            if not device:
                print(f"[BT] Device {address} not in discovered list")
                return False
            
            print(f"[BT] Creating BleakClient for {address} ({device.name})")
            print(f"[BT] Device info: RSSI={device.rssi}, Services={device.advertised_services}")
            
            # Create client with timeout hints for faster connection
            client = BleakClient(address, timeout=10.0)
            
            # Check if device is already system-connected
            system_connected = self._get_system_connected_devices()
            is_system_connected = address.upper() in system_connected
            
            if is_system_connected:
                print(f"[BT] Device is already connected at system level, using existing connection")
                print(f"[BT] Skipping Bleak connection (input via /dev/input/jsX)")
                # Don't call connect() for system-connected devices
                # Just track as connected and let joystick thread handle input
            else:
                print(f"[BT] Attempting connection to {address} (timeout: 10s)")
                try:
                    # Use a 10-second timeout on the actual connect call
                    await asyncio.wait_for(client.connect(), timeout=10.0)
                    print(f"[BT] Connection established")
                except asyncio.TimeoutError:
                    print(f"[BT] Connection timeout after 10 seconds - device may be out of range or not responding")
                    print(f"[BT] Device signal: RSSI={device.rssi} (range: -50 to -100 dBm)")
                    print(f"[BT] Troubleshooting:")
                    print(f"[BT]   - Ensure device is powered on and in pairing mode")
                    print(f"[BT]   - Check Bluetooth adapter: hciconfig, systemctl status bluetooth")
                    print(f"[BT]   - Try: bluetoothctl -> pair {address} -> connect {address}")
                    raise TimeoutError(f"Connection to {address} timed out - device not responding")
            
            self.connected_devices[address] = client
            self.connected_device_address = address  # Set as active input device
            
            # Update device state
            device.connected = True
            device.paired = True
            
            print(f"[BT] Successfully connected to {address}")
            
            # Only inspect GATT if we actually connected (not for system-connected devices)
            if not is_system_connected:
                # Inspect device and setup appropriate input method
                await self._inspect_device_gatt(client, address)
            
            self._emit_status("device_connected", {"address": address, "name": device.name})
            return True
        
        except asyncio.TimeoutError as e:
            error_msg = f"TimeoutError: Connection timeout - device not responding"
            print(f"[BT] Connection timeout: {error_msg}")
            self._emit_status("device_connect_failed", {"address": address, "error": error_msg})
            return False
        except BleakError as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Bleak error during connection: {error_msg}")
            self._emit_status("device_connect_failed", {"address": address, "error": error_msg})
            return False
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[BT] Connection error (unexpected): {error_msg}")
            print(f"[BT] Traceback:\n{tb}")
            self._emit_status("device_connect_failed", {"address": address, "error": error_msg})
            return False
    
    async def _inspect_device_gatt(self, client: BleakClient, address: str):
        """Inspect and log device's GATT services and characteristics."""
        try:
            print(f"\n[BT] ========== GATT PROFILE FOR {address} ==========")
            
            # Try to get services - method differs by Bleak version
            try:
                # Newer Bleak versions (0.19+)
                services = client.services
            except (AttributeError, TypeError):
                # Older Bleak versions
                services = await client.get_services()
            
            # Detect input method: vendor-specific (notifications) vs standard HID (polling)
            vendor_notify_chars = []
            standard_hid_available = False
            
            for service in services:
                service_uuid = service.uuid
                print(f"[BT] Service: {service_uuid}")
                
                for char in service.characteristics:
                    char_uuid = char.uuid
                    props = ", ".join(char.properties) if char.properties else "none"
                    print(f"[BT]   └─ Char: {char_uuid}")
                    print(f"[BT]      Properties: {props}")
                    
                    # Detect vendor-specific characteristics with notify
                    if "notify" in char.properties and "f000ff" in char_uuid.lower():
                        vendor_notify_chars.append(char_uuid)
                        print(f"[BT]      [VENDOR-SPECIFIC INPUT DETECTED]")
                    
                    # Detect standard HID Input Report
                    if char_uuid.lower() == "00002a4d-0000-1000-8000-00805f9b34fb":
                        standard_hid_available = True
                        print(f"[BT]      [STANDARD HID INPUT DETECTED]")
                    
                    # Try to read if it's readable and not too risky
                    if "read" in char.properties:
                        try:
                            # Skip reading non-essential characteristics to avoid timeouts
                            # Only read device info, battery, or vendor characteristics
                            skip_read = char_uuid.lower() in [
                                "00002b2a-0000-1000-8000-00805f9b34fb",  # GATT Database Hash
                            ]
                            
                            if not skip_read:
                                data = await asyncio.wait_for(client.read_gatt_char(char_uuid), timeout=1.0)
                                hex_str = " ".join(f"{b:02x}" for b in data[:20])
                                ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in data[:20])
                                print(f"[BT]      Data (read): {hex_str}")
                                if ascii_str.strip():
                                    print(f"[BT]      ASCII: {ascii_str}")
                        except asyncio.TimeoutError:
                            print(f"[BT]      Data (read timeout)")
                        except Exception as read_err:
                            print(f"[BT]      Data (read failed): {type(read_err).__name__}")
            
            # Setup input method based on what's available
            if vendor_notify_chars:
                print(f"[BT] Using NOTIFICATION mode (vendor-specific): {vendor_notify_chars}")
                await self._setup_notification_listeners(client, address, vendor_notify_chars)
            elif standard_hid_available:
                print(f"[BT] Using POLLING mode (standard HID)")
            else:
                print(f"[BT] WARNING: No recognized input characteristics found")
            
            print(f"[BT] ====================================================\n")
        
        except Exception as e:
            print(f"[BT] GATT inspection error: {type(e).__name__}: {e}")
    
    async def _setup_notification_listeners(self, client: BleakClient, address: str, char_uuids: List[str]):
        """Setup notification listeners on vendor-specific characteristics."""
        def notification_handler(char_uuid: str):
            """Factory to create a notification handler with captured uuid."""
            def handler(sender, data: bytearray):
                print(f"[BT] Notification on {char_uuid}: {data.hex()}")
                self.notification_data[char_uuid] = bytes(data)
            return handler
        
        try:
            for char_uuid in char_uuids:
                handler = notification_handler(char_uuid)
                
                # Start notifications
                await client.start_notify(char_uuid, handler)
                print(f"[BT] Started notifications on {char_uuid}")
                
                # Try to enable CCCD (Client Characteristic Configuration Descriptor)
                # Different Bleak versions use different methods
                try:
                    services = client.services
                    for service in services:
                        for char in service.characteristics:
                            if char.uuid.lower() == char_uuid.lower():
                                # Found the characteristic, look for its CCCD descriptor
                                for descriptor in char.descriptors:
                                    if descriptor.uuid.lower() == "00002902-0000-1000-8000-00805f9b34fb":
                                        # Try writing to CCCD using different methods
                                        try:
                                            # Method 1: Direct write with descriptor.handle (older Bleak)
                                            await client.write_gatt_descriptor(descriptor.handle, bytes([0x01, 0x00]))
                                            print(f"[BT] Enabled CCCD (handle) for {char_uuid}")
                                        except (AttributeError, TypeError):
                                            # Method 2: Use descriptor UUID directly (newer Bleak)
                                            await client.write_gatt_descriptor(descriptor.uuid, bytes([0x01, 0x00]))
                                            print(f"[BT] Enabled CCCD (uuid) for {char_uuid}")
                                        break
                except Exception as cccd_err:
                    print(f"[BT] CCCD write skipped: {type(cccd_err).__name__}")
                
                # Also try writing an enable command to the characteristic itself
                # (some vendor devices need this to activate input mode)
                try:
                    print(f"[BT] Writing enable command to {char_uuid}...")
                    # Try different enable commands - start with simple ones
                    for enable_cmd in [bytes([0x01]), bytes([0xFF]), bytes([0x01, 0x00])]:
                        try:
                            await client.write_gatt_char(char_uuid, enable_cmd, response=False)
                            print(f"[BT] Sent enable command: {enable_cmd.hex()}")
                            break
                        except:
                            pass
                except Exception as cmd_err:
                    print(f"[BT] Enable command write failed: {type(cmd_err).__name__} (may not be needed)")
        
        except Exception as e:
            print(f"[BT] Failed to setup notifications: {type(e).__name__}: {e}")
    
    def disconnect_device(self, address: str) -> bool:
        """Disconnect from a device."""
        try:
            print(f"[BT] Disconnecting from {address}...")
            result = self._run_async(self._disconnect_async(address))
            return result
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[BT] Disconnection error: {error_msg}")
            print(f"[BT] Traceback:\n{tb}")
            return False
    
    async def _disconnect_async(self, address: str) -> bool:
        """Async disconnect from device."""
        try:
            if address not in self.connected_devices:
                return False
            
            # Stop joystick reading if we're disconnecting the active device
            if self.connected_device_address == address:
                self.joystick_running = False
            
            client = self.connected_devices[address]
            try:
                await client.disconnect()
            except RuntimeError as e:
                if "different loop" in str(e):
                    # Event loop conflict - try to force close
                    print(f"[BT] Forcing disconnect due to event loop conflict")
                    try:
                        # Try to close without waiting for proper disconnect
                        if hasattr(client, '_close_client'):
                            client._close_client()
                    except:
                        pass
                else:
                    raise
            
            del self.connected_devices[address]
            
            # Clear active input device if this was it
            if self.connected_device_address == address:
                self.connected_device_address = None
            
            # Update device state
            if address in self.devices:
                self.devices[address].connected = False
            
            print(f"[BT] Successfully disconnected from {address}")
            self._emit_status("device_disconnected", {"address": address})
            return True
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if "different loop" not in error_msg:
                tb = traceback.format_exc()
                print(f"[BT] Disconnect error: {error_msg}")
                print(f"[BT] Traceback:\n{tb}")
            return False
    
    def remove_device(self, address: str) -> bool:
        """Remove/forget a device (disconnect first)."""
        try:
            # Disconnect first if connected
            if address in self.connected_devices:
                self.disconnect_device(address)
            
            # Remove from list
            if address in self.devices:
                del self.devices[address]
            
            print(f"[BT] Removed device {address}")
            self._emit_status("device_removed", {"address": address})
            return True
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[BT] Remove error: {error_msg}")
            print(f"[BT] Traceback:\n{tb}")
            return False
    
    def get_devices_list(self) -> List[Dict]:
        """Get list of discovered devices as dictionaries, including system-connected devices."""
        # Check system for actual connections
        system_connected = self._get_system_connected_devices()
        print(f"[BT] get_devices_list() called, system_connected={system_connected}")
        
        devices_list = []
        seen_addresses = set()
        
        # Add discovered devices (with updated connection status from system)
        for address, device in self.devices.items():
            seen_addresses.add(address.upper())
            device_dict = device.to_dict()
            # Mark as connected if it's actually connected at system level
            if address.upper() in system_connected:
                device_dict["connected"] = True
                print(f"[BT] Marking discovered device as connected: {address} → connected={device_dict.get('connected')}")
            else:
                print(f"[BT] Device {address} not in system_connected")
            devices_list.append(device_dict)
        
        # Add system-connected devices that weren't discovered
        print(f"[BT] Checking for system-only devices: system_connected={system_connected}, seen={seen_addresses}")
        for sys_address in system_connected:
            if sys_address not in seen_addresses:
                # Device is connected at system level but not in our discovered list
                # Query the system for the device name
                device_name = self._get_device_name_from_system(sys_address)
                if not device_name:
                    device_name = f"Connected Device ({sys_address[-5:]})"
                
                device = BluetoothDevice(sys_address, device_name, -100)
                device.connected = True
                device.paired = True
                
                # IMPORTANT: Store it in self.devices so _connect_async can find it
                self.devices[sys_address] = device
                
                device_dict = device.to_dict()
                devices_list.append(device_dict)
                print(f"[BT] Added system-connected device: {sys_address} ({device_name}) with connected={device_dict.get('connected')}")
        
        print(f"[BT] get_devices_list() returning {len(devices_list)} total devices (system_connected count: {len(system_connected)})")
        for i, d in enumerate(devices_list):
            print(f"[BT]   [{i}] {d.get('address')}: connected={d.get('connected')}")
        return devices_list
    
    def get_device(self, address: str) -> Optional[BluetoothDevice]:
        """Get a specific device by address. Checks system devices if not in cache."""
        # Check if already in cache
        if address in self.devices:
            return self.devices[address]
        
        # Check if it's a system-connected device
        system_connected = self._get_system_connected_devices()
        if address.upper() in system_connected:
            device_name = self._get_device_name_from_system(address)
            if not device_name:
                device_name = f"Connected Device ({address[-5:]})"
            
            device = BluetoothDevice(address, device_name, -100)
            device.paired = True
            device.connected = True
            self.devices[address] = device
            print(f"[BT] Found system-connected device: {address} ({device_name})")
            return device
        
        return None
    
    def read_characteristic(self, device_address: str, uuid: str) -> Optional[bytes]:
        """Read a characteristic value from a connected device."""
        try:
            result = self._run_async(self._read_characteristic_async(device_address, uuid))
            return result
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Read characteristic error: {error_msg}")
            return None
    
    async def _read_characteristic_async(self, device_address: str, uuid: str) -> Optional[bytes]:
        """Async read characteristic."""
        try:
            if device_address not in self.connected_devices:
                print(f"[BT] Device {device_address} not connected")
                return None
            
            client = self.connected_devices[device_address]
            value = await client.read_gatt_char(uuid)
            return value
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Read error: {error_msg}")
            return None
    
    def write_characteristic(self, device_address: str, uuid: str, data: bytes) -> bool:
        """Write a characteristic value to a connected device."""
        try:
            result = self._run_async(self._write_characteristic_async(device_address, uuid, data))
            return result
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Write characteristic error: {error_msg}")
            return False
    
    async def _write_characteristic_async(self, device_address: str, uuid: str, data: bytes) -> bool:
        """Async write characteristic."""
        try:
            if device_address not in self.connected_devices:
                print(f"[BT] Device {device_address} not connected")
                return False
            
            client = self.connected_devices[device_address]
            await client.write_gatt_char(uuid, data)
            return True
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Write error: {error_msg}")
            return False
    
    def cleanup(self):
        """Cleanup: disconnect all devices and stop scanning."""
        if self.scanning:
            self.stop_scan()
        
        # Disconnect all connected devices
        for address in list(self.connected_devices.keys()):
            self.disconnect_device(address)
    
    def get_connected_device(self) -> Optional[str]:
        """Get the address of the currently active input device."""
        return self.connected_device_address
    
    def read_controller_input(self) -> Dict:
        """
        Read input from connected device.
        Returns: {"pan_axis": float, "tilt_axis": float, "buttons": dict}
        
        pan_axis and tilt_axis are normalized values from -1.0 to 1.0
        """
        try:
            if not self.connected_device_address:
                return {"pan_axis": 0.0, "tilt_axis": 0.0, "buttons": {}}
            
            result = self._run_async(self._read_input_async(self.connected_device_address))
            if result:
                self.last_input_data = result
                return result
            return self.last_input_data
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Error reading input: {error_msg}")
            return self.last_input_data
    
    async def _read_input_async(self, device_address: str) -> Optional[Dict]:
        """
        Async read input from joystick device.
        Since the gamepad is system-connected, we read from /dev/input/jsX via the inputs library.
        """
        try:
            if device_address not in self.connected_devices:
                return None
            
            # Ensure joystick reading thread is running
            if not self.joystick_running:
                self._start_joystick_reading_thread()
            
            # Return the latest joystick state
            if self.last_joystick_input != self.last_input_data:
                result = self.last_joystick_input.copy()
                # Only log when values actually change
                if (abs(result.get('pan_axis', 0) - self.last_input_data.get('pan_axis', 0)) > 0.05 or
                    abs(result.get('tilt_axis', 0) - self.last_input_data.get('tilt_axis', 0)) > 0.05):
                    print(f"[INPUT] pan={result.get('pan_axis'):.2f}, tilt={result.get('tilt_axis'):.2f}")
                return result
            
            return self.last_joystick_input
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[INPUT] Read error: {error_msg}")
            return None
    
    def _start_joystick_reading_thread(self):
        """Start a background thread to read joystick input."""
        if not pygame:
            print("[INPUT] pygame library not available, cannot read joystick")
            return
        
        if self.joystick_running:
            return
        
        print("[INPUT] Starting joystick reading thread...")
        self.joystick_running = True
        self.joystick_thread = threading.Thread(target=self._joystick_read_loop, daemon=True)
        self.joystick_thread.start()
    
    def _find_gamepad_device(self):
        """Find the gamepad input device using pygame."""
        if not pygame:
            return None
        
        try:
            pygame.init()
            pygame.joystick.init()
            
            joystick_count = pygame.joystick.get_count()
            print(f"[INPUT] Found {joystick_count} joystick(s)")
            
            if joystick_count == 0:
                print("[INPUT] No joysticks detected")
                return None
            
            # Use the first joystick
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"[INPUT] Using joystick: {joystick.get_name()}")
            return joystick
        
        except Exception as e:
            print(f"[INPUT] Error initializing pygame: {type(e).__name__}: {e}")
            return None
    
    def _joystick_read_loop(self):
        """Background thread loop to read joystick events via pygame."""
        try:
            if not pygame:
                print("[INPUT] pygame not available")
                self.joystick_running = False
                return
            
            # Find and initialize pygame joystick
            joystick = self._find_gamepad_device()
            if not joystick:
                print("[INPUT] No gamepad device found")
                print("[INPUT] Try: jstest /dev/input/js0")
                self.joystick_running = False
                return
            
            print(f"[INPUT] Opened joystick: {joystick.get_name()}")
            print(f"[INPUT]   Axes: {joystick.get_numaxes()}")
            print(f"[INPUT]   Buttons: {joystick.get_numbuttons()}")
            print(f"[INPUT]   Hats: {joystick.get_numhats()}")
            
            # Track current state
            state = {
                "pan_axis": 0.0,     # Left stick X
                "tilt_axis": 0.0,    # Left stick Y
                "right_x": 0.0,      # Right stick X
                "right_y": 0.0,      # Right stick Y
                "buttons": {}        # Button states
            }
            
            print("[INPUT] Joystick reading thread started")
            
            # Create a clock for event processing timing (avoid blocking)
            clock = pygame.time.Clock()
            
            while self.joystick_running:
                try:
                    # Process pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.JOYAXISMOTION:
                            # Analog stick axes - normalize with deadzone
                            axis_map = {
                                0: "pan_axis",      # Left stick X
                                1: "tilt_axis",     # Left stick Y
                                2: "right_x",       # Right stick X
                                3: "right_y",       # Right stick Y
                            }
                            
                            if event.axis in axis_map:
                                # Apply deadzone
                                value = event.value
                                if abs(value) < 0.05:
                                    value = 0.0
                                
                                state[axis_map[event.axis]] = value
                        
                        elif event.type == pygame.JOYBUTTONDOWN:
                            button_map = {
                                0: "a",
                                1: "b",
                                2: "x",
                                3: "y",
                                4: "lb",
                                5: "rb",
                                6: "select",
                                7: "start",
                                8: "l_stick",
                                9: "r_stick",
                            }
                            if event.button in button_map:
                                state["buttons"][button_map[event.button]] = True
                        
                        elif event.type == pygame.JOYBUTTONUP:
                            button_map = {
                                0: "a",
                                1: "b",
                                2: "x",
                                3: "y",
                                4: "lb",
                                5: "rb",
                                6: "select",
                                7: "start",
                                8: "l_stick",
                                9: "r_stick",
                            }
                            if event.button in button_map:
                                state["buttons"][button_map[event.button]] = False
                        
                        elif event.type == pygame.JOYHATMOTION:
                            # D-pad (hat switch)
                            x, y = event.value
                            state["buttons"]["d_up"] = y > 0
                            state["buttons"]["d_down"] = y < 0
                            state["buttons"]["d_left"] = x < 0
                            state["buttons"]["d_right"] = x > 0
                    
                    # Update shared state
                    self.last_joystick_input = state.copy()
                    
                    # Limit event processing frequency to avoid busy-waiting
                    clock.tick(60)  # 60 Hz max
                
                except Exception as e:
                    if self.joystick_running:
                        print(f"[INPUT] Event processing error: {type(e).__name__}: {e}")
                    break
        
        except Exception as e:
            print(f"[INPUT] Joystick thread error: {type(e).__name__}: {e}")
        finally:
            print("[INPUT] Joystick reading thread stopped")
            try:
                pygame.quit()
            except:
                pass
            self.joystick_running = False
    
    def _parse_hid_input(self, data: bytes) -> Dict:
        """
        Parse HID input data from a Bluetooth controller.
        
        Generic HID format (simplified):
        - Bytes 0-1: Left stick X, Y (0-255, center at 128)
        - Bytes 2-3: Right stick X, Y (0-255, center at 128)
        - Byte 4: Buttons (bit flags)
        """
        if not data or len(data) < 5:
            return {"pan_axis": 0.0, "tilt_axis": 0.0, "buttons": {}}
        
        try:
            # Normalize stick values from 0-255 to -1.0 to 1.0
            left_x = (data[0] - 128) / 128.0  # Pan axis (horizontal)
            left_y = (data[1] - 128) / 128.0  # Tilt axis (vertical, inverted)
            
            # Clamp to -1.0 to 1.0
            left_x = max(-1.0, min(1.0, left_x))
            left_y = max(-1.0, min(1.0, left_y))
            
            # Simple dead zone: values close to 0 are treated as 0
            if abs(left_x) < 0.1:
                left_x = 0.0
            if abs(left_y) < 0.1:
                left_y = 0.0
            
            buttons = {}
            if len(data) > 4:
                buttons_byte = data[4]
                buttons = {
                    "a": bool(buttons_byte & 0x01),
                    "b": bool(buttons_byte & 0x02),
                    "x": bool(buttons_byte & 0x04),
                    "y": bool(buttons_byte & 0x08),
                }
            
            return {
                "pan_axis": left_x,
                "tilt_axis": left_y,
                "buttons": buttons
            }
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[BT] Parse error: {error_msg}")
            return {"pan_axis": 0.0, "tilt_axis": 0.0, "buttons": {}}
        
        print("[BT] Bluetooth controller cleaned up")

