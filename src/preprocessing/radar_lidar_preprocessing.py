import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from filterpy.kalman import KalmanFilter
import open3d as o3d

class RadarLidarPreprocessor:
    """
    Class for preprocessing radar and LIDAR data to determine field of view,
    detect objects, and calculate their distances and relative speeds.
    """
    def __init__(self, sensor_type='radar', max_range=100.0, angle_resolution=0.1, 
                 distance_resolution=0.1, velocity_resolution=0.1):
        """
        Initialize the radar/LIDAR preprocessor.
        
        Args:
            sensor_type (str): Type of sensor ('radar' or 'lidar').
            max_range (float): Maximum detection range in meters.
            angle_resolution (float): Angular resolution in degrees.
            distance_resolution (float): Distance resolution in meters.
            velocity_resolution (float): Velocity resolution in m/s (for radar only).
        """
        self.sensor_type = sensor_type.lower()
        self.max_range = max_range
        self.angle_resolution = angle_resolution
        self.distance_resolution = distance_resolution
        self.velocity_resolution = velocity_resolution
        
        # Initialize Kalman filter for tracking
        self.trackers = {}
        
        # Define field of view parameters (can be adjusted based on sensor specs)
        if self.sensor_type == 'radar':
            self.horizontal_fov = 60.0  # degrees
            self.vertical_fov = 20.0    # degrees
        else:  # LIDAR
            self.horizontal_fov = 360.0  # degrees
            self.vertical_fov = 30.0     # degrees
    
    def _initialize_kalman_filter(self):
        """
        Initialize a Kalman filter for object tracking.
        
        Returns:
            KalmanFilter: Initialized Kalman filter.
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]
        
        # State transition matrix (constant velocity model)
        dt = 0.1  # Time step
        kf.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(2) * 0.1
        
        # Process noise
        q = 0.01  # Process noise parameter
        kf.Q = np.array([
            [dt**4/4*q, dt**3/2*q, 0, 0],
            [dt**3/2*q, dt**2*q, 0, 0],
            [0, 0, dt**4/4*q, dt**3/2*q],
            [0, 0, dt**3/2*q, dt**2*q]
        ])
        
        # Initial state uncertainty
        kf.P *= 10
        
        return kf
    
    def preprocess_radar_data(self, raw_data):
        """
        Preprocess raw radar data to extract object information.
        
        Args:
            raw_data (numpy.ndarray): Raw radar data containing range, angle, and Doppler information.
            
        Returns:
            dict: Processed radar data with field of view and detected objects.
        """
        # In a real implementation, this would process actual radar data
        # Here we'll simulate the processing
        
        # Extract range, angle, and Doppler information
        ranges = raw_data[:, 0]  # Distance in meters
        angles = raw_data[:, 1]  # Angle in degrees
        dopplers = raw_data[:, 2]  # Doppler velocity in m/s
        
        # Determine field of view
        fov = {
            'horizontal_fov': self.horizontal_fov,
            'vertical_fov': self.vertical_fov,
            'max_range': self.max_range
        }
        
        # Detect objects using clustering or peak detection
        # Here we'll use a simple peak detection on the range profile
        peaks, _ = find_peaks(ranges, height=0.5, distance=10)
        
        # Extract object information
        objects = []
        for i, peak in enumerate(peaks):
            # Calculate object position in Cartesian coordinates
            angle_rad = np.radians(angles[peak])
            x = ranges[peak] * np.cos(angle_rad)
            y = ranges[peak] * np.sin(angle_rad)
            
            # Calculate relative velocity (negative means approaching)
            velocity = dopplers[peak]
            
            # Create object entry
            obj = {
                'id': i,
                'distance': float(ranges[peak]),
                'angle': float(angles[peak]),
                'position': [float(x), float(y)],
                'relative_speed': float(velocity),
                'type': self._classify_radar_object(ranges[peak], velocity)
            }
            objects.append(obj)
        
        # Track objects using Kalman filters
        tracked_objects = self._track_objects(objects)
        
        return {
            'field_of_view': fov,
            'objects': tracked_objects
        }
    
    def preprocess_lidar_data(self, point_cloud):
        """
        Preprocess LIDAR point cloud data to extract object information.
        
        Args:
            point_cloud (numpy.ndarray): Point cloud data with shape (N, 3) for (x, y, z) coordinates.
            
        Returns:
            dict: Processed LIDAR data with field of view and detected objects.
        """
        # In a real implementation, this would process actual LIDAR point cloud data
        # Here we'll simulate the processing using Open3D
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Determine field of view
        fov = {
            'horizontal_fov': self.horizontal_fov,
            'vertical_fov': self.vertical_fov,
            'max_range': self.max_range
        }
        
        # Segment ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
        ground_cloud = pcd.select_by_index(inliers)
        objects_cloud = pcd.select_by_index(inliers, invert=True)
        
        # Cluster objects
        clusters = objects_cloud.cluster_dbscan(eps=0.5, min_points=10)
        
        # Extract object information
        unique_clusters = set(clusters)
        objects = []
        
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get points belonging to this cluster
            cluster_indices = [j for j, c in enumerate(clusters) if c == cluster_id]
            cluster_points = np.asarray(objects_cloud.points)[cluster_indices]
            
            # Calculate centroid and dimensions
            centroid = np.mean(cluster_points, axis=0)
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            dimensions = max_bounds - min_bounds
            
            # Calculate distance from origin
            distance = np.linalg.norm(centroid[:2])  # XY distance
            
            # Calculate angle from origin
            angle = np.degrees(np.arctan2(centroid[1], centroid[0]))
            
            # Classify object based on dimensions
            obj_type = self._classify_lidar_object(dimensions)
            
            # Create object entry
            obj = {
                'id': i,
                'distance': float(distance),
                'angle': float(angle),
                'position': [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                'dimensions': [float(d) for d in dimensions],
                'type': obj_type,
                'relative_speed': None  # LIDAR doesn't provide velocity directly
            }
            objects.append(obj)
        
        return {
            'field_of_view': fov,
            'objects': objects,
            'ground_plane': plane_model
        }
    
    def _classify_radar_object(self, distance, velocity):
        """
        Classify radar-detected object based on its characteristics.
        
        Args:
            distance (float): Distance to the object in meters.
            velocity (float): Relative velocity of the object in m/s.
            
        Returns:
            str: Object classification.
        """
        # This is a simplified classification based on velocity patterns
        # In a real implementation, this would use more sophisticated methods
        
        if abs(velocity) < 0.5:
            return 'stationary_object'
        elif abs(velocity) < 2.0:
            return 'pedestrian'
        elif abs(velocity) < 5.0:
            return 'cyclist'
        elif abs(velocity) < 15.0:
            return 'vehicle'
        else:
            return 'fast_vehicle'
    
    def _classify_lidar_object(self, dimensions):
        """
        Classify LIDAR-detected object based on its dimensions.
        
        Args:
            dimensions (numpy.ndarray): Object dimensions [width, length, height].
            
        Returns:
            str: Object classification.
        """
        # This is a simplified classification based on object dimensions
        # In a real implementation, this would use more sophisticated methods
        
        width, length, height = dimensions
        volume = width * length * height
        
        if height < 0.5:
            return 'ground_object'
        elif height < 1.0 and volume < 1.0:
            return 'small_object'
        elif 1.0 < height < 2.0 and volume < 2.0:
            return 'pedestrian'
        elif 1.0 < height < 2.0 and 2.0 < volume < 5.0:
            return 'cyclist'
        elif 1.0 < height < 3.0 and volume > 5.0:
            return 'vehicle'
        elif height > 3.0:
            return 'large_vehicle'
        else:
            return 'unknown'
    
    def _track_objects(self, detected_objects):
        """
        Track objects across frames using Kalman filters.
        
        Args:
            detected_objects (list): List of detected objects.
            
        Returns:
            list: List of tracked objects with updated IDs and velocities.
        """
        # Update existing trackers and create new ones as needed
        current_ids = set()
        
        for obj in detected_objects:
            obj_id = obj['id']
            current_ids.add(obj_id)
            
            # Get position
            pos = np.array([obj['position'][0], obj['position'][1]])
            
            if obj_id in self.trackers:
                # Update existing tracker
                kf = self.trackers[obj_id]
                kf.predict()
                kf.update(pos)
                
                # Update object with filtered state
                state = kf.x
                obj['position'][0] = float(state[0])  # x position
                obj['position'][1] = float(state[2])  # y position
                
                # If this is LIDAR data (no velocity), estimate velocity from Kalman filter
                if obj['relative_speed'] is None:
                    velocity_x = float(state[1])  # x velocity
                    velocity_y = float(state[3])  # y velocity
                    obj['relative_speed'] = float(np.sqrt(velocity_x**2 + velocity_y**2))
                    
                    # Determine if object is approaching or moving away
                    if obj['position'][0] != 0 or obj['position'][1] != 0:
                        direction = np.array([obj['position'][0], obj['position'][1]])
                        direction = direction / np.linalg.norm(direction)
                        velocity = np.array([velocity_x, velocity_y])
                        if np.dot(direction, velocity) < 0:
                            obj['relative_speed'] *= -1  # Approaching
            else:
                # Create new tracker
                kf = self._initialize_kalman_filter()
                kf.x = np.array([pos[0], 0, pos[1], 0])  # Initial state
                kf.update(pos)
                self.trackers[obj_id] = kf
        
        # Remove trackers for objects that are no longer detected
        for obj_id in list(self.trackers.keys()):
            if obj_id not in current_ids:
                del self.trackers[obj_id]
        
        return detected_objects
    
    def process_data(self, data):
        """
        Process sensor data based on the sensor type.
        
        Args:
            data: Raw sensor data (format depends on sensor type).
            
        Returns:
            dict: Processed sensor data.
        """
        if self.sensor_type == 'radar':
            return self.preprocess_radar_data(data)
        else:  # LIDAR
            return self.preprocess_lidar_data(data)
    
    def visualize_field_of_view(self, processed_data, show_objects=True):
        """
        Visualize the field of view and detected objects.
        
        Args:
            processed_data (dict): Processed sensor data.
            show_objects (bool): Whether to show detected objects.
            
        Returns:
            matplotlib.figure.Figure: The visualization figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get field of view parameters
        fov = processed_data['field_of_view']
        max_range = fov['max_range']
        h_fov = fov['horizontal_fov']
        
        # Draw field of view
        angles = np.linspace(-h_fov/2, h_fov/2, 100)
        x = max_range * np.cos(np.radians(angles))
        y = max_range * np.sin(np.radians(angles)