import torch
import torchvision
from torchvision.ops import roi_pool

# Assuming you have a feature map with shape [batch_size, channels, height, width]
feature_map = torch.randn(1, 256, 64, 64)

# Assuming you have a tensor representing region of interests (ROIs)
# The ROIs tensor should have shape [num_rois, 5], where each row represents [batch_index, x_min, y_min, x_max, y_max]
rois = torch.tensor([[0, 10, 10, 30, 30], [0, 40, 40, 60, 60]])

# Perform ROI pooling
output_size = (7, 7)  # Output size of the pooled feature map
pooled_features = roi_pool(feature_map, rois, output_size)

# The pooled_features tensor will have shape [num_rois, channels, output_size[0], output_size[1]]
print(pooled_features.shape)
