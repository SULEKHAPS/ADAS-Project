import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class SensorFusion:
    """
    Class for correlating objects detected by camera and radar/LIDAR sensors,
    performing sensor fusion, and identifying potential obstructions.
    """
    def __init__(self, camera_fov=60.0, max_correlation_distance=5.0):
        """
        Initialize the sensor fusion module.
        
        Args:
            camera_fov (float): Camera field of view in degrees.
            max_correlation_distance (float): Maximum distance for object correlation in meters.
        """
        self.camera_fov = camera_fov
        self.max_correlation_distance = max_correlation_distance
        
        # Initialize Kalman filter for tracking fused objects
        self.trackers = {}
        
        # Store previous fused objects for tracking
        self.prev_fused_objects = []
    
    def _initialize_kalman_filter(self):
        """
        Initialize a Kalman filter for object tracking.
        
        Returns:
            KalmanFilter: Initialized Kalman filter.
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, vx, y, vy, z, vz], Measurement: [x, y, z]
        
        # State transition matrix (constant velocity model)
        dt = 0.1  # Time step
        kf.F = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(3) * 0.1
        
        # Process noise
        q = 0.01  # Process noise parameter
        kf.Q = np.eye(6) * q
        
        # Initial state uncertainty
        kf.P *= 10
        
        return kf
    
    def correlate_objects(self, camera_objects, radar_lidar_objects):
        """
        Correlate objects detected by camera and radar/LIDAR sensors.
        
        Args:
            camera_objects (list): Objects detected by the camera.
            radar_lidar_objects (list): Objects detected by radar or LIDAR.
            
        Returns:
            list: Correlated objects with combined information.
        """
        # If either list is empty, return the non-empty list or an empty list if both are empty
        if not camera_objects:
            return self._convert_radar_lidar_objects(radar_lidar_objects)
        if not radar_lidar_objects:
            return self._convert_camera_objects(camera_objects)
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = np.zeros((len(camera_objects), len(radar_lidar_objects)))
        
        for i, cam_obj in enumerate(camera_objects):
            for j, rl_obj in enumerate(radar_lidar_objects):
                # Calculate cost based on angular position (since camera doesn't provide distance)
                # This is a simplified approach; in a real implementation, we would use more sophisticated methods
                cost = self._calculate_correlation_cost(cam_obj, rl_obj)
                cost_matrix[i, j] = cost
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create correlated objects
        correlated_objects = []
        used_camera_indices = set(row_ind)
        used_radar_lidar_indices = set(col_ind)
        
        # Add correlated objects
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.max_correlation_distance:
                correlated_obj = self._fuse_objects(camera_objects[i], radar_lidar_objects[j])
                correlated_objects.append(correlated_obj)
        
        # Add uncorrelated camera objects
        for i in range(len(camera_objects)):
            if i not in used_camera_indices:
                obj = self._convert_camera_object(camera_objects[i])
                correlated_objects.append(obj)
        
        # Add uncorrelated radar/LIDAR objects
        for j in range(len(radar_lidar_objects)):
            if j not in used_radar_lidar_indices:
                obj = self._convert_radar_lidar_object(radar_lidar_objects[j])
                correlated_objects.append(obj)
        
        return correlated_objects
    
    def _calculate_correlation_cost(self, camera_obj, radar_lidar_obj):
        """
        Calculate the cost for correlating a camera object with a radar/LIDAR object.
        
        Args:
            camera_obj (dict): Camera-detected object.
            radar_lidar_obj (dict): Radar/LIDAR-detected object.
            
        Returns:
            float: Correlation cost (lower is better).
        """
        # In a real implementation, this would use more sophisticated methods
        # Here we'll use a simple angular difference
        
        # Estimate camera object angle (assuming object is in center of image if no bounding box)
        if 'bounding_box' in camera_obj and camera_obj['bounding_box']:
            # If we have a bounding box, estimate angle based on horizontal position
            bbox = camera_obj['bounding_box']
            center_x = (bbox[0] + bbox[2]) / 2  # Center x-coordinate of bounding box
            image_width = 1.0  # Normalized image width
            normalized_x = (center_x / image_width) - 0.5  # -0.5 to 0.5
            cam_angle = normalized_x * self.camera_fov  # Convert to angle
        else:
            # If no bounding box, assume object is in center of image
            cam_angle = 0.0
        
        # Get radar/LIDAR object angle
        rl_angle = radar_lidar_obj['angle']
        
        # Calculate angular difference
        angle_diff = abs(cam_angle - rl_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Calculate cost based on angular difference
        cost = angle_diff
        
        # If object types are available and don't match, increase cost
        if 'class_name' in camera_obj and 'type' in radar_lidar_obj:
            if not self._object_types_compatible(camera_obj['class_name'], radar_lidar_obj['type']):
                cost += 100  # Large penalty for incompatible types
        
        return cost
    
    def _object_types_compatible(self, camera_type, radar_lidar_type):
        """
        Check if camera and radar/LIDAR object types are compatible.
        
        Args:
            camera_type (str): Camera object type.
            radar_lidar_type (str): Radar/LIDAR object type.
            
        Returns:
            bool: True if types are compatible, False otherwise.
        """
        # Define compatibility mapping
        compatibility = {
            'car': ['vehicle', 'fast_vehicle'],
            'truck': ['vehicle', 'large_vehicle'],
            'pedestrian': ['pedestrian'],
            'cyclist': ['cyclist'],
            'motorcycle': ['cyclist', 'vehicle'],
            'bus': ['large_vehicle', 'vehicle'],
            'traffic_sign': ['stationary_object', 'small_object'],
            'traffic_light': ['stationary_object', 'small_object'],
            'animal': ['pedestrian', 'small_object'],
            'obstacle': ['stationary_object', 'ground_object', 'small_object']
        }
        
        # Check if radar/LIDAR type is compatible with camera type
        if camera_type in compatibility:
            return radar_lidar_type in compatibility[camera_type]
        
        return False
    
    def _fuse_objects(self, camera_obj, radar_lidar_obj):
        """
        Fuse information from camera and radar/LIDAR objects.
        
        Args:
            camera_obj (dict): Camera-detected object.
            radar_lidar_obj (dict): Radar/LIDAR-detected object.
            
        Returns:
            dict: Fused object with combined information.
        """
        # Create fused object with information from both sensors
        fused_obj = {
            'id': radar_lidar_obj['id'],  # Use radar/LIDAR ID for tracking
            'position': radar_lidar_obj['position'],  # Use radar/LIDAR position (more accurate)
            'distance': radar_lidar_obj['distance'],  # Use radar/LIDAR distance
            'angle': radar_lidar_obj['angle'],  # Use radar/LIDAR angle
            'relative_speed': radar_lidar_obj.get('relative_speed'),  # Use radar/LIDAR speed if available
            'class_name': camera_obj.get('class_name', 'unknown'),  # Use camera class (more accurate for classification)
            'confidence': camera_obj.get('confidence', 1.0),  # Use camera confidence
            'bounding_box': camera_obj.get('bounding_box'),  # Use camera bounding box
            'dimensions': radar_lidar_obj.get('dimensions'),  # Use radar/LIDAR dimensions if available
            'sensor_sources': ['camera', 'radar_lidar'],  # Mark as fused from both sources
            'fusion_confidence': 1.0  # High confidence for fused objects
        }
        
        return fused_obj
    
    def _convert_camera_object(self, camera_obj):
        """
        Convert a camera object to the standard format.
        
        Args:
            camera_obj (dict): Camera-detected object.
            
        Returns:
            dict: Converted object.
        """
        # Create object with camera information only
        obj = {
            'id': camera_obj.get('id', -1),  # Use camera ID if available, otherwise -1
            'position': None,  # Camera doesn't provide position
            'distance': None,  # Camera doesn't provide distance
            'angle': None,  # Camera doesn't provide angle
            'relative_speed': None,  # Camera doesn't provide speed
            'class_name': camera_obj.get('class_name', 'unknown'),
            'confidence': camera_obj.get('confidence', 1.0),
            'bounding_box': camera_obj.get('bounding_box'),
            'dimensions': None,  # Camera doesn't provide dimensions
            'sensor_sources': ['camera'],  # Mark as from camera only
            'fusion_confidence': 0.5  # Lower confidence for single-source objects
        }
        
        return obj
    
    def _convert_radar_lidar_object(self, radar_lidar_obj):
        """
        Convert a radar/LIDAR object to the standard format.
        
        Args:
            radar_lidar_obj (dict): Radar/LIDAR-detected object.
            
        Returns:
            dict: Converted object.
        """
        # Map radar/LIDAR object type to camera class name
        type_to_class = {
            'vehicle': 'car',
            'fast_vehicle': 'car',
            'large_vehicle': 'truck',
            'pedestrian': 'pedestrian',
            'cyclist': 'cyclist',
            'stationary_object': 'obstacle',
            'ground_object': 'obstacle',
            'small_object': 'obstacle'
        }
        
        # Create object with radar/LIDAR information only
        obj = {
            'id': radar_lidar_obj['id'],
            'position': radar_lidar_obj['position'],
            'distance': radar_lidar_obj['distance'],
            'angle': radar_lidar_obj['angle'],
            'relative_speed': radar_lidar_obj.get('relative_speed'),
            'class_name': type_to_class.get(radar_lidar_obj.get('type', ''), 'unknown'),
            'confidence': 0.7,  # Default confidence for radar/LIDAR objects
            'bounding_box': None,  # Radar/LIDAR doesn't provide bounding box
            'dimensions': radar_lidar_obj.get('dimensions'),
            'sensor_sources': ['radar_lidar'],  # Mark as from radar/LIDAR only
            'fusion_confidence': 0.7  # Medium confidence for single-source objects
        }
        
        return obj
    
    def _convert_camera_objects(self, camera_objects):
        """
        Convert a list of camera objects to the standard format.
        
        Args:
            camera_objects (list): List of camera-detected objects.
            
        Returns:
            list: List of converted objects.
        """
        return [self._convert_camera_object(obj) for obj in camera_objects]
    
    def _convert_radar_lidar_objects(self, radar_lidar_objects):
        """
        Convert a list of radar/LIDAR objects to the standard format.
        
        Args:
            radar_lidar_objects (list): List of radar/LIDAR-detected objects.
            
        Returns:
            list: List of converted objects.
        """
        return [self._convert_radar_lidar_object(obj) for obj in radar_lidar_objects]
    
    def track_objects(self, fused_objects):
        """
        Track objects across frames using Kalman filters.
        
        Args:
            fused_objects (list): List of fused objects from current frame.
            
        Returns:
            list: List of tracked objects with updated IDs and velocities.
        """
        # Update existing trackers and create new ones as needed
        current_ids = set()
        
        for obj in fused_objects:
            # Skip objects without position information
            if obj['position'] is None:
                continue
                
            obj_id = obj['id']
            current_ids.add(obj_id)
            
            # Get position
            if len(obj['position']) >= 3:
                pos = np.array([obj['position