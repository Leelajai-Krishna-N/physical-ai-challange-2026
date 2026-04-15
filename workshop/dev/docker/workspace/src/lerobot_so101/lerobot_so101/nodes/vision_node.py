import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs # Import the module, not the specific function

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Parameters for object detection (Default: Red object)
        self.declare_parameter('target_color_lower', [0, 100, 100]) # HSV
        self.declare_parameter('target_color_upper', [10, 255, 255]) # HSV

        # Camera Intrinsics (Default D435i approx values for 640x480)
        self.declare_parameter('fx', 600.0)
        self.declare_parameter('fy', 600.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        self.lower_color = np.array(self.get_parameter('target_color_lower').value, dtype=np.uint8)
        self.upper_color = np.array(self.get_parameter('target_color_upper').value, dtype=np.uint8)
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        self.bridge = CvBridge()

        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State to hold the latest depth image
        self.latest_depth_image = None

        # Subscriber to the camera feeds
        self.image_sub = self.create_subscription(
            Image,
            '/d435i/image_raw',
            self.image_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            '/d435i/depth_raw',
            self.depth_callback,
            10)

        # Publisher for the object coordinates relative to base_link
        self.publisher = self.create_publisher(Point, '/detected_object_base', 10)

        self.get_logger().info('Vision Node started. Projecting detections to base_link...')

    def depth_callback(self, msg):
        # Store the latest depth image as a numpy array
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        if self.latest_depth_image is None:
            self.get_logger().warn('Waiting for depth image...')
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest_contour) > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 1. Get depth at the center pixel
                        depth = self.latest_depth_image[cy, cx]

                        if np.isnan(depth) or depth <= 0:
                            self.get_logger().info('Object detected but depth is invalid.')
                            return

                        # 2. Project 2D pixel to 3D camera frame (d435i_link)
                        x_cam = (cx - self.cx) * depth / self.fx
                        y_cam = (cy - self.cy) * depth / self.fy
                        z_cam = depth

                        # 3. Transform from d435i_link to base_link
                        try:
                            # Look up transform from base_link to d435i_link
                            transform = self.tf_buffer.lookup_transform(
                                'base_link',
                                'd435i_link',
                                rclpy.time.Time())

                            # Create a point in camera frame
                            point_cam = Point()
                            point_cam.x = x_cam
                            point_cam.y = y_cam
                            point_cam.z = z_cam

                            # Transform the point to base_link frame
                            point_base = tf2_geometry_msgs.do_transform_translation(point_cam, transform)

                            self.publisher.publish(point_base)
                            self.get_logger().info(f'Object detected! Base frame: x={point_base.x:.3f}, y={point_base.y:.3f}, z={point_base.z:.3f}')

                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                            self.get_logger().error(f'TF Transform failed: {str(e)}')

            else:
                self.get_logger().info('Object not found in view.')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
