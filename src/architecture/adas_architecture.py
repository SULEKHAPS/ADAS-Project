import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_adas_architecture_diagram():
    """
    Creates a system-level flowchart for ADAS operation showing the integration
    of various sensors and machine learning models.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Define colors
    sensor_color = '#3498db'  # Blue
    processing_color = '#2ecc71'  # Green
    fusion_color = '#9b59b6'  # Purple
    decision_color = '#e74c3c'  # Red
    output_color = '#f39c12'  # Orange
    arrow_color = '#34495e'  # Dark blue
    
    # Define box properties
    box_width = 0.2
    box_height = 0.1
    
    # Draw sensor layer
    sensors = [
        {'name': 'Camera', 'x': 0.2, 'y': 0.9},
        {'name': 'Radar', 'x': 0.5, 'y': 0.9},
        {'name': 'LIDAR', 'x': 0.8, 'y': 0.9}
    ]
    
    for sensor in sensors:
        rect = patches.Rectangle((sensor['x'], sensor['y']), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor=sensor_color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(sensor['x'] + box_width/2, sensor['y'] + box_height/2, sensor['name'], 
               ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw preprocessing layer
    preprocessing = [
        {'name': 'Image Processing', 'x': 0.2, 'y': 0.75},
        {'name': 'Radar Signal\nProcessing', 'x': 0.5, 'y': 0.75},
        {'name': 'Point Cloud\nProcessing', 'x': 0.8, 'y': 0.75}
    ]
    
    for process in preprocessing:
        rect = patches.Rectangle((process['x'], process['y']), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor=processing_color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(process['x'] + box_width/2, process['y'] + box_height/2, process['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw object detection layer
    detection = [
        {'name': 'CNN Object\nDetection', 'x': 0.2, 'y': 0.6},
        {'name': 'Radar Object\nTracking', 'x': 0.5, 'y': 0.6},
        {'name': 'LIDAR Object\nSegmentation', 'x': 0.8, 'y': 0.6}
    ]
    
    for detect in detection:
        rect = patches.Rectangle((detect['x'], detect['y']), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor=processing_color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(detect['x'] + box_width/2, detect['y'] + box_height/2, detect['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw sensor fusion layer
    fusion_rect = patches.Rectangle((0.4, 0.45), 0.2, 0.1, 
                                 linewidth=1, edgecolor='black', facecolor=fusion_color, alpha=0.8)
    ax.add_patch(fusion_rect)
    ax.text(0.5, 0.5, 'Sensor Fusion\n(Kalman Filter)', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw decision making layer
    decision_modules = [
        {'name': 'Collision\nPrediction', 'x': 0.3, 'y': 0.3},
        {'name': 'Path\nPlanning', 'x': 0.5, 'y': 0.3},
        {'name': 'Risk\nAssessment', 'x': 0.7, 'y': 0.3}
    ]
    
    for module in decision_modules:
        rect = patches.Rectangle((module['x'], module['y']), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor=decision_color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(module['x'] + box_width/2, module['y'] + box_height/2, module['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw output actions layer
    actions = [
        {'name': 'Warning\nAlerts', 'x': 0.2, 'y': 0.15},
        {'name': 'Automatic\nBraking', 'x': 0.4, 'y': 0.15},
        {'name': 'Lane Keep\nAssist', 'x': 0.6, 'y': 0.15},
        {'name': 'Adaptive\nCruise Control', 'x': 0.8, 'y': 0.15}
    ]
    
    for action in actions:
        rect = patches.Rectangle((action['x'], action['y']), box_width, box_height, 
                               linewidth=1, edgecolor='black', facecolor=output_color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(action['x'] + box_width/2, action['y'] + box_height/2, action['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows connecting the layers
    # Sensors to preprocessing
    for i in range(len(sensors)):
        ax.arrow(sensors[i]['x'] + box_width/2, sensors[i]['y'], 0, -0.05, 
                head_width=0.01, head_length=0.01, fc=arrow_color, ec=arrow_color, linewidth=1.5)
    
    # Preprocessing to object detection
    for i in range(len(preprocessing)):
        ax.arrow(preprocessing[i]['x'] + box_width/2, preprocessing[i]['y'], 0, -0.05, 
                head_width=0.01, head_length=0.01, fc=arrow_color, ec=arrow_color, linewidth=1.5)
    
    # Object detection to sensor fusion
    for i in range(len(detection)):
        ax.arrow(detection[i]['x'] + box_width/2, detection[i]['y'], 
                (0.5 - (detection[i]['x'] + box_width/2))/2, -0.05, 
                head_width=0.01, head_length=0.01, fc=arrow_color, ec=arrow_color, linewidth=1.5, 
                shape='right', length_includes_head=True)
    
    # Sensor fusion to decision modules
    ax.arrow(0.5, 0.45, 0, -0.05, 
            head_width=0.01, head_length=0.01, fc=arrow_color, ec=arrow_color, linewidth=1.5)
    
    # Decision modules to actions
    for i in range(len(decision_modules)):
        for j in range(len(actions)):
            ax.arrow(decision_modules[i]['x'] + box_width/2, decision_modules[i]['y'], 
                    (actions[j]['x'] + box_width/2 - (decision_modules[i]['x'] + box_width/2))/2, -0.05, 
                    head_width=0.01, head_length=0.01, fc=arrow_color, ec=arrow_color, linewidth=0.5, 
                    alpha=0.3, shape='right', length_includes_head=True)
    
    # Add layer labels
    ax.text(0.05, 0.9 + box_height/2, 'SENSORS', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(0.05, 0.75 + box_height/2, 'PREPROCESSING', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(0.05, 0.6 + box_height/2, 'OBJECT DETECTION', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(0.05, 0.5, 'SENSOR FUSION', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(0.05, 0.3 + box_height/2, 'DECISION MAKING', fontsize=12, fontweight='bold', ha='right', va='center')
    ax.text(0.05, 0.15 + box_height/2, 'ACTIONS', fontsize=12, fontweight='bold', ha='right', va='center')
    
    # Add title
    ax.set_title('ADAS System Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=sensor_color, edgecolor='black', alpha=0.8, label='Sensor Components'),
        patches.Patch(facecolor=processing_color, edgecolor='black', alpha=0.8, label='Processing Components'),
        patches.Patch(facecolor=fusion_color, edgecolor='black', alpha=0.8, label='Fusion Components'),
        patches.Patch(facecolor=decision_color, edgecolor='black', alpha=0.8, label='Decision Components'),
        patches.Patch(facecolor=output_color, edgecolor='black', alpha=0.8, label='Action Components')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              fancybox=True, shadow=True, ncol=5)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('adas_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ADAS architecture diagram created and saved as 'adas_architecture.png'")

if __name__ == "__main__":
    create_adas_architecture_diagram()