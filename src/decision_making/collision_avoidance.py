import numpy as np
import matplotlib.pyplot as plt
import time

class CollisionAvoidance:
    """
    Class for detecting potential collisions and implementing avoidance strategies
    based on fused sensor data.
    """
    def __init__(self, time_to_collision_threshold=3.0, distance_threshold=10.0, 
                 warning_distance=30.0, critical_distance=15.0, vehicle_width=2.0, 
                 vehicle_length=4.5, lane_width=3.5):
        """
        Initialize the collision avoidance system.
        
        Args:
            time_to_collision_threshold (float): Minimum time to collision for alert in seconds.
            distance_threshold (float): Minimum distance for collision alert in meters.
            warning_distance (float): Distance at which to issue a warning in meters.
            critical_distance (float): Distance at which to take corrective action in meters.
            vehicle_width (float): Width of the vehicle in meters.
            vehicle_length (float): Length of the vehicle in meters.
            lane_width (float): Width of a standard lane in meters.
        """
        self.ttc_threshold = time_to_collision_threshold
        self.distance_threshold = distance_threshold
        self.warning_distance = warning_distance
        self.critical_distance = critical_distance
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.lane_width = lane_width
        
        # Define vehicle safety envelope (rectangle around vehicle)
        self.safety_envelope = {
            'front': vehicle_length * 1.5,  # 1.5x vehicle length in front
            'rear': vehicle_length * 0.5,   # 0.5x vehicle length behind
            'sides': vehicle_width * 0.75    # 0.75x vehicle width on each side
        }
        
        # Store previous alerts to prevent alert flooding
        self.previous_alerts = {}
        self.alert_cooldown = 2.0  # seconds
        
        # Store vehicle state
        self.vehicle_state = {
            'speed': 0.0,            # m/s
            'acceleration': 0.0,      # m/s²
            'heading': 0.0,           # degrees
            'steering_angle': 0.0,    # degrees
            'lane_position': 0.0      # meters from lane center (0 = center)
        }
        
        # Store previous collision avoidance actions
        self.previous_actions = []
    
    def update_vehicle_state(self, speed, acceleration=None, heading=None, 
                           steering_angle=None, lane_position=None):
        """
        Update the current vehicle state.
        
        Args:
            speed (float): Current vehicle speed in m/s.
            acceleration (float, optional): Current vehicle acceleration in m/s².
            heading (float, optional): Current vehicle heading in degrees.
            steering_angle (float, optional): Current steering angle in degrees.
            lane_position (float, optional): Current position in lane in meters from center.
        """
        self.vehicle_state['speed'] = speed
        
        if acceleration is not None:
            self.vehicle_state['acceleration'] = acceleration
        if heading is not None:
            self.vehicle_state['heading'] = heading
        if steering_angle is not None:
            self.vehicle_state['steering_angle'] = steering_angle
        if lane_position is not None:
            self.vehicle_state['lane_position'] = lane_position
    
    def detect_potential_collisions(self, fused_objects):
        """
        Detect potential collisions with objects detected by sensors.
        
        Args:
            fused_objects (list): List of objects from sensor fusion.
            
        Returns:
            list: List of potential collision objects with collision information.
        """
        collision_risks = []
        current_time = time.time()
        
        for obj in fused_objects:
            # Skip objects without position or distance information
            if obj['position'] is None or obj['distance'] is None:
                continue
            
            # Calculate time to collision (TTC) if relative speed is available
            ttc = float('inf')  # Default to infinity
            if obj['relative_speed'] is not None and obj['relative_speed'] < 0:  # Negative means approaching
                # Convert to positive for calculation
                approach_speed = -obj['relative_speed']
                if approach_speed > 0:
                    ttc = obj['distance'] / approach_speed
            
            # Check if object is in front of the vehicle (simplified)
            # In a real implementation, we would use more sophisticated methods
            is_in_path = self._is_object_in_path(obj)
            
            # Determine collision risk
            if is_in_path and (ttc < self.ttc_threshold or obj['distance'] < self.distance_threshold):
                risk_level = self._calculate_risk_level(obj, ttc)
                
                collision_obj = {
                    'object': obj,
                    'time_to_collision': ttc,
                    'risk_level': risk_level,
                    'recommended_action': self._determine_avoidance_action(obj, ttc, risk_level)
                }
                
                collision_risks.append(collision_obj)
        
        # Sort by risk level (highest first) and then by TTC (lowest first)
        collision_risks.sort(key=lambda x: (-x['risk_level'], x['time_to_collision']))
        
        return collision_risks
    
    def _is_object_in_path(self, obj):
        """
        Determine if an object is in the vehicle's path.
        
        Args:
            obj (dict): Object from sensor fusion.
            
        Returns:
            bool: True if object is in path, False otherwise.
        """
        # In a real implementation, this would use more sophisticated methods
        # considering the vehicle's trajectory, lane information, etc.
        
        # Simple check: object is roughly in front of the vehicle and within lane width
        if len(obj['position']) >= 2:
            x, y = obj['position'][0], obj['position'][1]
            
            # Assuming vehicle is at origin facing positive x direction
            # Check if object is in front and within lane width
            in_front = x > 0
            within_lane = abs(y) < (self.lane_width / 2 + self.safety_envelope['sides'])
            
            return in_front and within_lane
        
        return False
    
    def _calculate_risk_level(self, obj, ttc):
        """
        Calculate the risk level of a potential collision.
        
        Args:
            obj (dict): Object from sensor fusion.
            ttc (float): Time to collision in seconds.
            
        Returns:
            int: Risk level (0-3, where 3 is highest).
        """
        # Risk levels:
        # 0: No risk
        # 1: Low risk - Warning
        # 2: Medium risk - Prepare for action
        # 3: High risk - Immediate action required
        
        distance = obj['distance']
        
        if distance < self.critical_distance or ttc < 1.0:
            return 3  # High risk
        elif distance < self.warning_distance or ttc < 2.0:
            return 2  # Medium risk
        elif distance < self.warning_distance * 1.5 or ttc < self.ttc_threshold:
            return 1  # Low risk
        else:
            return 0  # No risk
    
    def _determine_avoidance_action(self, obj, ttc, risk_level):
        """
        Determine the appropriate avoidance action based on collision risk.
        
        Args:
            obj (dict): Object from sensor fusion.
            ttc (float): Time to collision in seconds.
            risk_level (int): Risk level (0-3).
            
        Returns:
            dict: Recommended avoidance action.
        """
        # Default action: no action
        action = {
            'type': 'none',
            'parameters': {}
        }
        
        # Determine action based on risk level
        if risk_level == 1:  # Low risk
            action = {
                'type': 'warning',
                'parameters': {
                    'message': f"Warning: {obj['class_name']} ahead",
                    'audio': True,
                    'visual': True
                }
            }
        elif risk_level == 2:  # Medium risk
            action = {
                'type': 'prepare_braking',
                'parameters': {
                    'message': f"Caution: {obj['class_name']} approaching",
                    'audio': True,
                    'visual': True,
                    'haptic': True,
                    'deceleration': 2.0  # m/s²
                }
            }
        elif risk_level == 3:  # High risk
            # Check if lane change is possible
            if self._is_lane_change_possible(obj):
                action = {
                    'type': 'lane_change',
                    'parameters': {
                        'message': f"Avoiding {obj['class_name']}",
                        'audio': True,
                        'visual': True,
                        'haptic': True,
                        'direction': 'left' if obj['position'][1] > 0 else 'right',
                        'deceleration': 3.0  # m/s²
                    }
                }
            else:
                action = {
                    'type': 'emergency_braking',
                    'parameters': {
                        'message': f"Emergency: {obj['class_name']} collision imminent",
                        'audio': True,
                        'visual': True,
                        'haptic': True,
                        'deceleration': 7.0  # m/s²
                    }
                }
        
        return action
    
    def _is_lane_change_possible(self, obj):
        """
        Determine if a lane change is possible to avoid collision.
        
        Args:
            obj (dict): Object from sensor fusion.
            
        Returns:
            bool: True if lane change is possible, False otherwise.
        """
        # In a real implementation, this would check for objects in adjacent lanes,
        # lane markings, road boundaries, etc.
        
        # Simplified implementation: always return False (emergency braking)
        # In a real system, this would be much more sophisticated
        return False
    
    def execute_avoidance_action(self, action):
        """
        Execute the recommended avoidance action.
        
        Args:
            action (dict): Avoidance action to execute.
            
        Returns:
            dict: Result of the action execution.
        """
        # In a real implementation, this would interface with vehicle control systems
        # Here we'll just simulate the action
        
        action_type = action['type']
        parameters = action['parameters']
        
        # Store the action
        self.previous_actions.append({
            'time': time.time(),
            'action': action
        })
        
        # Simulate action execution
        result = {
            'action': action_type,
            'success': True,
            'message': f"Executed {action_type}"
        }
        
        # In a real implementation, we would update vehicle state based on the action
        if action_type == 'emergency_braking':
            # Simulate emergency braking
            deceleration = parameters.get('deceleration', 7.0)
            # Update vehicle state (simplified)
            new_speed = max(0, self.vehicle_state['speed'] - deceleration * 0.1)  # Assuming 0.1s time step
            self.update_vehicle_state(speed=new_speed)
        elif action_type == 'prepare_braking':
            # Simulate gentle braking
            deceleration = parameters.get('deceleration', 2.0)
            # Update vehicle state (simplified)
            new_speed = max(0, self.vehicle_state['speed'] - deceleration * 0.1)  # Assuming 0.1s time step
            self.update_vehicle_state(speed=new_speed)
        elif action_type == 'lane_change':
            # Simulate lane change
            direction = parameters.get('direction', 'left')
            # Update vehicle state (simplified)
            lane_offset = 1.0 if direction == 'left' else -1.0
            self.update_vehicle_state(lane_position=lane_offset)
        
        return result
    
    def process_sensor_data(self, fused_objects):
        """
        Process sensor data to detect and avoid potential collisions.
        
        Args:
            fused_objects (list): List of objects from sensor fusion.
            
        Returns:
            dict: Collision avoidance results.
        """
        # Detect potential collisions
        collision_risks = self.detect_potential_collisions(fused_objects)
        
        # Execute avoidance actions for high-risk collisions
        executed_actions = []
        for risk in collision_risks:
            if risk['risk_level'] > 0:  # Only execute actions for actual risks
                result = self.execute_avoidance_action(risk['recommended_action'])
                executed_actions.append({
                    'risk': risk,
                    'result': result
                })
                
                # Only execute the highest priority action
                break
        
        return {
            'collision_risks': collision_risks,
            'executed_actions': executed_actions
        }
    
    def visualize_collision_risks(self, fused_objects, collision_risks):
        """
        Visualize collision risks and avoidance actions.
        
        Args:
            fused_objects (list): List of objects from sensor fusion.
            collision_risks (list): List of potential collision objects.
            
        Returns:
            matplotlib.figure.Figure: The visualization figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set axis limits
        ax.set_xlim(-20, 60)
        ax.set_ylim(-20, 20)
        
        # Draw vehicle (simplified as a rectangle)
        vehicle_rect = plt.Rectangle((-self.vehicle_length/2, -self.vehicle_width/2), 
                                   self.vehicle_length, self.vehicle_width, 
                                   color='blue', alpha=0.7)
        ax.add_patch(vehicle_rect)
        
        # Draw safety envelope
        safety_rect = plt.Rectangle((-self.vehicle_length/2 - self.safety_envelope['rear'], 
                                  -self.vehicle_width/2 - self.safety_envelope['sides']), 
                                 self