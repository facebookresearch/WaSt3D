import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_enc', self.positional_encoding(d_model, max_len))

    def forward(self, x):
        return x + self.pos_enc

    def positional_encoding(self, d_model, max_len):
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros(max_len, d_model)
        print("pos shape: ", pos.shape)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc


# def positional_encoding_3d(d_model, num_coordinates):
#     """
#     Add positional encoding to 3D coordinates.

#     Args:
#     - d_model (int): The dimension of the positional encoding.
#     - num_coordinates (int): The number of 3D coordinate sets.

#     Returns:
#     - torch.Tensor: A tensor containing positional encodings for 3D coordinates.
#     """
#     pos = torch.arange(0, num_coordinates).unsqueeze(1)  # Shape: (num_coordinates, 1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))  # Shape: (d_model / 2, )
#     pos_enc = torch.zeros(num_coordinates, d_model)  # Shape: (num_coordinates, d_model)

#     # Compute positional encodings
#     pos_enc[:, 0::2] = torch.sin(pos * div_term)
#     pos_enc[:, 1::2] = torch.cos(pos * div_term)

#     return pos_enc



# def positional_embedding_3d(coordinates, d):
#     """
#     Add positional embedding to a batch of 3D coordinates.

#     Args:
#     - coordinates (torch.Tensor): A tensor of 3D coordinates of shape [batch_size, 3].
#     - d (int): The dimensionality of the positional embedding.

#     Returns:
#     - torch.Tensor: A tensor containing positional embeddings of shape [batch_size, 3 * 2 * d].
#     """
#     batch_size = coordinates.size(0)

#     # Generate positional embeddings for each coordinate in the batch
#     pos = torch.arange(0, 3).repeat(batch_size, 1)  # Shape: [batch_size, 3]

#     div_term = torch.exp(torch.arange(0, d * 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d * 2)))  # Shape: [d * 2, ]

#     # Compute positional embeddings
#     pos_embeddings = torch.zeros(batch_size, 3 * 2 * d)  # Shape: [batch_size, 3 * 2 * d]
#     pos_embeddings[:, 0::2] = torch.sin(pos * div_term)
#     pos_embeddings[:, 1::2] = torch.cos(pos * div_term)

#     return pos_embeddings



class Embedder:
    """
    Positional embedding module used in NERFs.
    Taken from here  https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py .
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)





class NeRFPositionalEncoding(nn.Module):
    def __init__(self, num_input_features, max_freq_log2=10, num_freqs=6):
        super(NeRFPositionalEncoding, self).__init__()

        self.num_input_features = num_input_features
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.num_features = num_input_features * 2  # Each input feature has a sine and cosine component

        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs).to("cuda")
        self.freq_bands = self.freq_bands.unsqueeze(0)  # Shape: [1, num_freqs]

    def forward(self, input_locations):
        # input_locations: Batch of 3D coordinates, shape [batch_size, num_input_features, 3]

        # Expand to match the number of frequencies
        input_locations = input_locations.unsqueeze(2)  # Shape: [batch_size, num_input_features, 1, 3]

        # Compute positional encoding
        sin_features = torch.sin(self.freq_bands * input_locations)
        cos_features = torch.cos(self.freq_bands * input_locations)

        # Concatenate sine and cosine components
        encoding = torch.cat([sin_features, cos_features], dim=-1)  # Shape: [batch_size, num_input_features, 1, num_features]

        return encoding

# # Example usage:
# batch_size = 4  # Adjust the batch size as needed
# num_input_features = 10  # Number of input features (positions) per location

# # Create a random batch of 3D coordinates
# input_locations = torch.rand((batch_size, num_input_features, 3))



# # Compute positional encoding
# positional_encoding = positional_encoder(input_locations)

# print(positional_encoding.shape)  # Should print [batch_size, num_input_features, 1, num_features]







class SphereProjectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len=100):
        super(SphereProjectionModel, self).__init__()
        self.hidden_dim = hidden_dim
        #self.positional_embedder = Embedder(input_dims=3, include_input=True)

        # Create the positional encoding layer
        num_input_features=3
        num_freqs=2 # was 8
        max_freq_log2=2 # was 10
        self.positional_encoder = NeRFPositionalEncoding(num_input_features=num_input_features,
                                                        max_freq_log2=max_freq_log2,
                                                        num_freqs=num_freqs)

        # MLP layers for encoding
        self.encoder = nn.Sequential(
            nn.Linear(num_input_features*num_freqs*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Linear layer for rotation matrix
        self.rotation_matrix_layer = nn.Linear(hidden_dim, 9)  # 3x3 rotation matrix

        # Decoder MLP layers for normalized 3D points
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_points):
        print("Input points shape: ", input_points.shape)

        # Apply positional encodings
        encoded_points = self.positional_encoder(input_points)

        print("encoded points shape: ", encoded_points.shape)
        encoded_points = torch.reshape(encoded_points, (-1, encoded_points.shape[1] * encoded_points.shape[2]))
        print("encoded points shape after concatenating: ", encoded_points.shape)

        # Encoding
        encoded = self.encoder(encoded_points)
        print("encoded shape: ", encoded.shape)

        # Predict rotation matrix
        rotation_matrix = self.rotation_matrix_layer(encoded)
        rotation_matrix = rotation_matrix.view(-1, 3, 3)  # Reshape to 3x3

        # Decode normalized 3D points
        normalized_points = self.decoder(encoded)
        print("normalized_points shape: ", normalized_points.shape)

        return normalized_points#, rotation_matrix




# class SphereProjectionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len=100):
#         super(SphereProjectionModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         #self.positional_embedder = Embedder(input_dims=3, include_input=True)

#         # Create the positional encoding layer
#         num_input_features=3
#         num_freqs=2 # was 8
#         max_freq_log2=2 # was 10
#         self.positional_encoder = NeRFPositionalEncoding(num_input_features=num_input_features,
#                                                         max_freq_log2=max_freq_log2,
#                                                         num_freqs=num_freqs)

#         # MLP layers for encoding
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim, bias=False),
#             nn.ReLU()
#         )

#         # Linear layer for rotation matrix
#         self.rotation_matrix_layer = nn.Linear(hidden_dim, 9)  # 3x3 rotation matrix

#         # Decoder MLP layers for normalized 3D points
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim, bias=False)
#         )

#     def forward(self, input_points):
#         print("Input points shape: ", input_points.shape)


#         # Encoding
#         encoded = self.encoder(input_points)
#         print("encoded shape: ", encoded.shape)

#         # Predict rotation matrix
#         rotation_matrix = self.rotation_matrix_layer(encoded)
#         rotation_matrix = rotation_matrix.view(-1, 3, 3)  # Reshape to 3x3

#         # Decode normalized 3D points
#         normalized_points = self.decoder(encoded)
#         print("normalized_points shape: ", normalized_points.shape)

#         return normalized_points#, rotation_matrix
