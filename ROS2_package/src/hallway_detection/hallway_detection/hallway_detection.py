"""
Publishes goal_poses for the nubot that will cause it to explore the environment. When the robot gets close to a wall, it will turn

PUBLISHERS:
  + binary_image (Image) - The binary image from the Deep Learning model
  

SUBSCRIBERS:
  + camera/XXXXX (Image) - The image from the camera on the robot

"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
from cv_bridge import CvBridge
import cv2
import torchvision
import torchvision.transforms as transforms
# from sensor_msgs.msg import Image


class Autoencoder(torch.nn.Module):

  def __init__(self, n_classes):
    super(Autoencoder, self).__init__()
    # encoder layers
    self.enc_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv1_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn1 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool1 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu1 = torch.nn.ReLU()
    self.enc_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool2 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu2 = torch.nn.ReLU()
    self.enc_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_bn3 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool3 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu3 = torch.nn.ReLU()
    self.enc_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_conv4_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_bn4 = torch.nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool4 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu4 = torch.nn.ReLU()
    # decoder layers
    self.dec_up1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv1 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_bn1 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu1 = torch.nn.ReLU()
    self.dec_up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv2 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu2 = torch.nn.ReLU()
    self.dec_up3 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn3 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu3 = torch.nn.ReLU()
    self.dec_up4 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding='same')
    self.dec_conv4_2 = torch.nn.Conv2d(in_channels=3, out_channels=n_classes, kernel_size=5, stride=1, padding='same')
    self.dec_bn4 = torch.nn.BatchNorm2d(num_features=n_classes, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu4 = torch.nn.ReLU()
    #self.dec_soft = torch.nn.Softmax(dim=1)

  def forward(self, x):
    # encoder forward pass
    a = self.enc_conv1(x)
    a = self.enc_conv1_2(a)
    a = self.enc_bn1(a)
    a = self.enc_relu1(a)
    res1 = self.enc_pool1(a)

    a = self.enc_conv2(res1)
    a = self.enc_conv2_2(a)
    a = self.enc_bn2(a)
    a = self.enc_relu2(a)
    res2 = self.enc_pool2(a)

    a = self.enc_conv3(res2)
    a = self.enc_conv3_2(a)
    a = self.enc_bn3(a)
    a = self.enc_relu3(a)
    res3 = self.enc_pool3(a)

    a = self.enc_conv4(res3)
    a = self.enc_conv4_2(a)
    a = self.enc_bn4(a)
    a = self.enc_relu4(a)
    res4 = self.enc_pool4(a)

    # decoder forward pass
    a = self.dec_up1(res4)
    a = self.dec_conv1(a)
    a = self.dec_conv1_2(a)
    a = self.dec_bn1(a)
    a = self.dec_relu1(a)

    a = torch.cat((a,res3),1)

    a = self.dec_up2(a)
    a = self.dec_conv2(a)
    a = self.dec_conv2_2(a)
    a = self.dec_bn2(a)
    a = self.dec_relu2(a)

    a = torch.cat((a,res2),1)

    a = self.dec_up3(a)
    a = self.dec_conv3(a)
    a = self.dec_conv3_2(a)
    a = self.dec_bn3(a)
    a = self.dec_relu3(a)

    a = torch.cat((a,res1),1)

    a = self.dec_up4(a)
    a = self.dec_conv4(a)
    a = self.dec_conv4_2(a)
    a = self.dec_bn4(a)
    a = self.dec_relu4(a)

    return a #self.dec_soft(a)

class RecurrentDepthBackbone(torch.nn.Module):
    def __init__(self, base_backbone) -> None:
        super().__init__()
        activation = torch.nn.ELU()
        last_activation = torch.nn.Tanh()
        self.base_backbone = base_backbone
        # if env_cfg == None:
        self.combination_mlp = torch.nn.Sequential(
                                    torch.nn.Linear(32 + 53, 128),
                                    activation,
                                    torch.nn.Linear(128, 32)
                                )
        # else:
        #     self.combination_mlp = torch.nn.Sequential(
        #                                 torch.nn.Linear(32 + 53, 128),
        #                                 activation,
        #                                 torch.nn.Linear(128, 32)
                                    # )
        self.rnn = torch.nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = torch.nn.Sequential(
                                torch.nn.Linear(512, 32),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class DepthOnlyFCBackbone58x87(torch.nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = torch.nn.ELU()
        self.image_compression = torch.nn.Sequential(
            # [1, 58, 87]
            torch.nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            torch.nn.Flatten(),
            # [32, 25, 39]
            torch.nn.Linear(64 * 25 * 39, 128),
            activation,
            torch.nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = torch.nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent

class Hallway_Detection(Node):
    def __init__(self):
        """
        Initializes the Hallway_Detection node
        """
        super().__init__('hallway_detection')

        self.declare_parameter("rate", 30.)
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        self.current_image = None
        # Create goal_pose publisher
        self.image_out_pub = self.create_publisher(Image, 'binary_image', 10)
        self.latent_pub = self.create_publisher(Image, 'latent', 10)
        self.encoded_latent_pub = self.create_publisher(Image, 'encoded_latent', 10)   
        self.red_green_pub = self.create_publisher(Image, 'red_green', 10)
        
        
        # Create laser scan subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # Create timer
        rate_seconds = 1 / self.rate
        self.timer = self.create_timer(rate_seconds, self.timer_callback)
        # self.model = Autoencoder(2)
        # self.model.load_state_dict(torch.load('/home/sdalal/ws/deep_learning_final/model'))
        # self.model.to(self.device)
        log_pth = "/home/sdalal/ws/winter_project/src/isaacgym/test/extreme-parkour/legged_gym/logs/parkour_new/go1-dist-2.0.7/traced/go1-dist-2.0.7-1900-base_jit.pt"
        import os
        model = os.path.join(log_pth)
        self.policy = torch.jit.load(model)
        self.depth_enc_state_dict = torch.load("/home/sdalal/ws/winter_project/src/isaacgym/test/extreme-parkour/legged_gym/logs/parkour_new/go1-dist-2.0.7/traced/go1-dist-2.0.7-1900-vision_weight.pt")
        self.depth_backbone = DepthOnlyFCBackbone58x87(1, 32, 512)
        self.depth_backbone.eval()
        self.depth_backbone.to(self.device)
        self.depth_encoders = RecurrentDepthBackbone(self.depth_backbone)
        # self.depth_encoders.load_state_dict(self.depth_enc_state_dict)
        self.resize_transform = torchvision.transforms.Resize((58, 87), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.latent = None


    def timer_callback(self):
        if self.current_image is not None:
            mod_image = self.process_depth_image(self.current_image)
            # self.get_logger().error(str(mod_image.unsqueeze(0).unsqueeze(0).shape))
            img2 = self.depth_backbone(mod_image.unsqueeze(0))
            self.latent = self.depth_encoders(mod_image.unsqueeze(0), torch.ones(1, 53))
            # self.get_logger().error(str(self.latent.shape))
            # image_tensor = self.ros2_image_to_tensor(self.current_image)
            # output_tensor = self.model_inference(self.latent, self.policy)[0].argmax(dim=0)
            output_tensor = self.model_inference(self.latent, self.policy)
            self.get_logger().error(str(output_tensor))
            
            # output_image = self.tensor_to_mask_image(output_tensor)
            # self.get_logger().error(str(output_tensor))
            output_image = self.tensor_to_ros2_image(img2)
            # # output_image = output_image.detach().cpu()
            # # red_green_image = self.apply_binary_filter_to_image(self.current_image, output_image)
            # # self.get_logger().error(output_image)
            self.image_out_pub.publish(output_image)
            # self.red_green_pub.publish(red_green_image)
        

    def image_callback(self, image):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        self.current_image = transform(cv_image)
        
        # image_tensor = self.ros2_image_to_tensor(self.current_image)
        # print(image_tensor)

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - 2.0) / (0.0- 2.0)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += 0.0 * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, 0.0, 2.0)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]
    
    def ros2_image_to_tensor(self, ros_image: Image) -> torch.Tensor:
        """
        Converts a ROS2 image message to a PyTorch tensor with normalized values (0 to 1).

        :param ros_image: The ROS2 image message to be converted.
        :return: A PyTorch tensor representing the image, with pixel values normalized to [0, 1].
        """
        bridge = CvBridge()
        
        # Convert ROS2 image to OpenCV image
        try:
            cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting ROS2 image to OpenCV: {e}")
            return None
        
        height, width, _ = cv_image.shape
        center_x, center_y = width // 2, height // 2
        half_width, half_height = 16, 16  # Half of the desired width and height
        
        cv_image = cv_image[center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

        # Crop the image to 480x480
        # cv_image = cv_image[0:720, 180:900]
        cv_image = cv2.resize(cv_image, (32, 32))
        
        # Convert the image to grayscale
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Convert the OpenCV grayscale image to a PyTorch tensor and normalize it
        image_tensor = torch.from_numpy(cv_image).view(-1)[:32].unsqueeze(0).float()  # Flatten and take first 32 elements
        # image_tensor = torch.from_numpy(cv_image).float()
        
        # Normalize the tensor to [0, 1] by dividing by the max value (255 for 8-bit images)
        image_tensor /= 255.0

        return image_tensor

    
    # convert tensor to ros2 image
    def tensor_to_ros2_image(self, tensor: torch.Tensor) -> Image:
        """
        Converts a PyTorch tensor to a ROS2 Image message.

        :param tensor: The PyTorch tensor to be converted.
        :return: A ROS2 Image message representing the tensor.
        """
        bridge = CvBridge()

        # Convert the PyTorch tensor to a NumPy array
        
        tensor = tensor * 255.0
        tensor = tensor.cpu()
        image_np = tensor.detach().numpy().astype(np.uint8)
        

        # Convert the NumPy array to a ROS2 Image message
        try:
            image_msg = bridge.cv2_to_imgmsg(image_np, encoding='mono8')
        except Exception as e:
            print(f"Error converting NumPy array to ROS2 Image: {e}")
            return None

        return image_msg

    # Use a trained model to take in an image tensor and convert the image to the output of the model
    def model_inference(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Uses a trained model to perform inference on an image tensor.

        :param image_tensor: The input image tensor to be processed.
        :param model: The PyTorch model to be used for inference.
        :return: The output tensor from the model.
        """
        # Set the model to evaluation mode
        # model.eval()

        # Perform inference
        # self.get_logger().error(str(image_tensor))
        num_envs = 1
        n_proprio = 3 + 2 + 3 + 4 + 36 + 4 +1
        num_scan = 132
        n_priv_explicit = 3 + 3 + 3
        n_priv_latent = 4 + 1 + 12 +12
        history_len = 10
        
        
        obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio)
        self.get_logger().error(str(image_tensor.shape))
        # image_tensor = image_tensor.reshape(1, )
        output = model(obs_input, image_tensor)
        # output = obs_input
        self.get_logger().error(str(output))
        return output
        
    def tensor_to_mask_image(self, tensor: torch.Tensor) -> Image:
        """
        Converts a PyTorch tensor to a ROS2 Image message.

        :param tensor: The PyTorch tensor to be converted.
        :return: A ROS2 Image message representing the tensor.
        """
        bridge = CvBridge()

        # Convert the PyTorch tensor to a NumPy array
        image_np = tensor.numpy()

        # Convert the NumPy array to a ROS2 Image message
        try:
            image_msg = bridge.cv2_to_imgmsg(image_np, encoding='passthrough')
        except Exception as e:
            print(f"Error converting NumPy array to ROS2 Image: {e}")
            return None

        return image_msg

def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Hallway_Detection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()