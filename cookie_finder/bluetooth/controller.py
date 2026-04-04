"""
Bluetooth device scanning and connection management using bleak.
Async-friendly BLE (Bluetooth Low Energy) controller for Orange Pi and other platforms.
"""

import asyncio
import threading
from typing import List, Dict, Optional, Callable
from bleak import BleakScanner, BleakClient, BleakError


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
        self.scan_thread: Optional[threading.Thread] = None
        self.status_callback: Optional[Callable] = None
    
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
    
    def _run_async(self, coro):
        """Helper to run async code from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule as task
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=15)
            else:
                return asyncio.run(coro)
        except RuntimeError:
            # No event loop in this thread, create new one
            return asyncio.run(coro)
    
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
            print(f"[BT] Scan error: {e}")
            self.scanning = False
            self._emit_status("scan_error", {"error": str(e)})
    
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
            print(f"[BT] Scan error: {e}")
            self.scanning = False
            self._emit_status("scan_error", {"error": str(e)})
    
    def connect_device(self, address: str) -> bool:
        """Attempt to connect to a device."""
        try:
            print(f"[BT] Connecting to {address}...")
            result = self._run_async(self._connect_async(address))
            return result
        except Exception as e:
            print(f"[BT] Connection error: {e}")
            self._emit_status("device_connect_error", {"address": address, "error": str(e)})
            return False
    
    async def _connect_async(self, address: str) -> bool:
        """Async connect to device."""
        try:
            # Check if already connected
            if address in self.connected_devices:
                return True
            
            # Create and connect client
            device = self.devices.get(address)
            if not device:
                print(f"[BT] Device {address} not in discovered list")
                return False
            
            client = BleakClient(address)
            await client.connect()
            self.connected_devices[address] = client
            
            # Update device state
            device.connected = True
            device.paired = True
            
            print(f"[BT] Successfully connected to {address}")
            self._emit_status("device_connected", {"address": address, "name": device.name})
            return True
        
        except BleakError as e:
            print(f"[BT] Connection failed: {e}")
            self._emit_status("device_connect_failed", {"address": address, "error": str(e)})
            return False
    
    def disconnect_device(self, address: str) -> bool:
        """Disconnect from a device."""
        try:
            print(f"[BT] Disconnecting from {address}...")
            result = self._run_async(self._disconnect_async(address))
            return result
        except Exception as e:
            print(f"[BT] Disconnection error: {e}")
            return False
    
    async def _disconnect_async(self, address: str) -> bool:
        """Async disconnect from device."""
        try:
            if address not in self.connected_devices:
                return False
            
            client = self.connected_devices[address]
            await client.disconnect()
            del self.connected_devices[address]
            
            # Update device state
            if address in self.devices:
                self.devices[address].connected = False
            
            print(f"[BT] Successfully disconnected from {address}")
            self._emit_status("device_disconnected", {"address": address})
            return True
        
        except Exception as e:
            print(f"[BT] Disconnection error: {e}")
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
            print(f"[BT] Remove error: {e}")
            return False
    
    def get_devices_list(self) -> List[Dict]:
        """Get list of discovered devices as dictionaries."""
        return [device.to_dict() for device in self.devices.values()]
    
    def get_device(self, address: str) -> Optional[BluetoothDevice]:
        """Get a specific device by address."""
        return self.devices.get(address)
    
    def read_characteristic(self, device_address: str, uuid: str) -> Optional[bytes]:
        """Read a characteristic value from a connected device."""
        try:
            result = self._run_async(self._read_characteristic_async(device_address, uuid))
            return result
        except Exception as e:
            print(f"[BT] Read characteristic error: {e}")
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
            print(f"[BT] Read error: {e}")
            return None
    
    def write_characteristic(self, device_address: str, uuid: str, data: bytes) -> bool:
        """Write a characteristic value to a connected device."""
        try:
            result = self._run_async(self._write_characteristic_async(device_address, uuid, data))
            return result
        except Exception as e:
            print(f"[BT] Write characteristic error: {e}")
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
            print(f"[BT] Write error: {e}")
            return False
    
    def cleanup(self):
        """Cleanup: disconnect all devices and stop scanning."""
        if self.scanning:
            self.stop_scan()
        
        # Disconnect all connected devices
        for address in list(self.connected_devices.keys()):
            self.disconnect_device(address)
        
        print("[BT] Bluetooth controller cleaned up")

