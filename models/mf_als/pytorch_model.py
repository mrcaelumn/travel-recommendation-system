import torch

# Define loss function
def loss_function(rating_input, rating_pred, user_embeddings, hotel_embeddings, reg_lambda):
    loss = torch.nn.functional.mse_loss(rating_pred, rating_input)
    reg_loss = reg_lambda * (torch.sum(torch.square(user_embeddings)) + torch.sum(torch.square(hotel_embeddings)))
    return loss + reg_loss

# Define PyTorch model
class MatrixFactorizationModel(torch.nn.Module):
    def __init__(self, num_users, num_hotels, num_latent_factors):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embeddings = torch.nn.Embedding(num_users, num_latent_factors)
        self.hotel_embeddings = torch.nn.Embedding(num_hotels, num_latent_factors)
        self.user_biases = torch.nn.Embedding(num_users, 1)
        self.hotel_biases = torch.nn.Embedding(num_hotels, 1)

    def forward(self, user_id_input, hotel_id_input):
        user_embedding = self.user_embeddings(user_id_input)
        hotel_embedding = self.hotel_embeddings(hotel_id_input)
        user_bias = self.user_biases(user_id_input)
        hotel_bias = self.hotel_biases(hotel_id_input)
        rating_pred = torch.sum(torch.mul(user_embedding, hotel_embedding), dim=1) + user_bias.squeeze() + hotel_bias.squeeze()
        return rating_pred