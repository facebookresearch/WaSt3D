import torch
import torch.nn.functional as F
device="cuda:0"
def debug_groups_loss():

    # Define a loss function
    def loss_function(pairwise_diff_matrices_pred, pairwise_diff_matrices_target=None):
        loss = torch.tensor(0.0, device=device)

        for k, matrix in enumerate(pairwise_diff_matrices_pred):
            if pairwise_diff_matrices_target is not None:
                diff = torch.square(matrix - pairwise_diff_matrices_target[k])  # Compute difference from target matrix (e.g., zeros)
            else:
                diff = torch.square(matrix)  # Compute difference from target matrix (e.g., zeros)
            loss += torch.mean(diff)  # Accumulate the loss

        return loss

    # Example usage:
    N = 8  # Number of elements
    d = 3  # Dimensionality
    K = 2  # Number of clusters

    # Generate random data and cluster indices for testing
    data = torch.rand(N, d, device='cuda', requires_grad=True)  # Place data on the GPU and require gradients
    print("initial data:")
    print(data)
    indices = torch.randint(1, K + 1, (N,), device='cuda')
    for k in range(1, K+1):
        print("cluster", k, ":")
        group_indices = (indices == k).nonzero().view(-1)
        print(data[group_indices])

    # Compute target matrices (e.g., all zeros)
    target_matrices = [torch.zeros(N, N, device='cuda') for _ in range(K)]

    # Create the optimizer
    optimizer = torch.optim.Adam([data], lr=0.01)

    # Number of optimization steps
    num_steps = 200

    # Optimization loop
    for step in range(num_steps):
        pairwise_diff_matrices = compute_pairwise_differences(data, indices)
        loss = loss_function(pairwise_diff_matrices, pairwise_diff_matrices_target=[x*0. + 1 for x in pairwise_diff_matrices])

        # Print loss for each pairwise_diff_matrix
        for k, matrix in enumerate(pairwise_diff_matrices):
            # print(f"Loss in Cluster {k + 1} at Step {step + 1}: {F.mse_loss(matrix, target_matrices[k])}")
            if step % 25== 0:
                print(f"Loss in Cluster {k + 1} at Step {step + 1}: {loss}")

        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Optimized data tensor:")
    print(data)
    for k in range(1, K+1):
        print("cluster", k, ":")
        group_indices = (indices == k).nonzero().view(-1)
        print(data[group_indices])


def compute_pairwise_differences(tensor, indices):
    """
    Group elements based on indices and compute pairwise differences within each group.

    Args:
    - tensor (torch.Tensor): Input tensor of shape [N, d], where N is the number of elements and d is the dimensionality.
    - indices (torch.Tensor or list): List of indices of shape [N] with elements in the range [1, ..., K].

    Returns:
    - list of torch.Tensor: A list of K square matrices containing pairwise differences within each group.
    """
    device = tensor.device
    K = torch.max(indices).item()  # Calculate the number of clusters

    # Initialize a list to store pairwise differences for each group
    pairwise_diff_matrices = []

    for k in range(1, K + 1):
         # Select elements that belong to cluster k
        group_indices = (indices == k).nonzero().view(-1)
        group_elements = tensor[group_indices]

        # Compute pairwise differences within the group (vectorized version)
        pairwise_diff = torch.cdist(group_elements, group_elements, p=2)  # Compute L2 distances

        pairwise_diff_matrices.append(pairwise_diff)

    return pairwise_diff_matrices



def debug_optimization_based_on_pairwise_differences():
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    import numpy as np

    def load_and_optimize_image(image_path, output_path, num_steps=1000, learning_rate=0.01, resize=None, device="cuda"):
        # Load the original image
        pil_image = Image.open(image_path)
        #pil_image = ImageOps.grayscale(pil_image)

        # Resize the image if specified
        if resize is not None:
            pil_image = pil_image.resize(resize)

        width, height = pil_image.size

        # Convert the image to a PyTorch tensor
        image_tensor = torch.Tensor(np.array(pil_image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Create a target matrix by computing pairwise differences in pixel values
        print("image_tensor.shape: ", image_tensor.shape)


        image_tensor = image_tensor.view(-1, 3, height*width)
        target_D = image_tensor[:, :, :, None] - image_tensor[:, :, None, :]
        target_D = target_D.detach().clone().to(device)

        # Initialize a random image of the same size as a PyTorch parameter
        optimized_image = torch.zeros(1, 3, height, width, requires_grad=True, device=device)# +torch.tensor(1., device=device)


        # Create an optimizer
        optimizer = optim.Adam([optimized_image], lr=learning_rate)

        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute pairwise differences for the optimized image
            x = optimized_image.view(-1, 3, height*width)
            D = x[:, :, :, None] - x[:, :, None, :]

            loss = torch.sum(torch.square(target_D - D))
            # Perform a gradient step
            loss.backward()
            optimizer.step()

            # Clip pixel values to [0, 1] to keep the image in the valid range
            optimized_image.data = torch.clamp(optimized_image.data, 0, 1)

            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{num_steps}, Loss: {loss.item()}")

        # Save the optimized image

        optimized_image = optimized_image.squeeze().detach().cpu().numpy()
        optimized_image = np.transpose(optimized_image, (1, 2, 0))
        optimized_image = (optimized_image * 255).astype(np.uint8)
        optimized_image = Image.fromarray(optimized_image)
        optimized_image.save(output_path)


    # Example usage:
    input_image_path = "VinylFence.jpg"
    output_image_path = "VinylFence_optimized.jpg"
    resize_dimensions = (64, 64)  # Set to None to keep the original size

    load_and_optimize_image(input_image_path, output_image_path, num_steps=1000, learning_rate=0.01, resize=resize_dimensions)

#debug_groups_loss()

debug_optimization_based_on_pairwise_differences()
