import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        #print('t_s',t_s)
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        t_s1 = (t_s-1) / (T - 1) 
        
        beta_t = beta_1 + (beta_T - beta_1) * t_s1  # Linear interpolation ？
        
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        sqrt_alpha_t = torch.sqrt(alpha_t)
        oneover_sqrt_alpha = 1 / sqrt_alpha_t

        ####
        t_s = t_s.item()
        t_range = torch.arange(0, t_s , dtype=torch.float32) / (T - 1)
        
        # Linear interpolation to calculate beta_t at each value in t_range
        beta_t_range = beta_1 + (beta_T - beta_1) * t_range
        
        # Calculate corresponding alpha_t values
        alpha_t_range = 1 - beta_t_range
        
        # Calculate cumulative product of alpha_t up to t_s
        

        alpha_t_bar = torch.cumprod(alpha_t_range, dim=0)[-1]  # Take the last value as it represents the product up to t_s
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)


        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }






 
    def forward(self, images, conditions):
        T = self.dmconfig.T
        device = images.device
        
        conditions_onehot = F.one_hot(conditions, num_classes=10).float().to(device)
        dropout_rate = self.dmconfig.mask_p 
        mask = torch.bernoulli(torch.full((conditions_onehot.size(0), 1), 1 - dropout_rate)).to(device)
        conditions_dropped = torch.where(mask == 1, conditions_onehot, torch.tensor(-1.0, device=device))
        
        t = torch.randint(1, T+1, (images.size(0),), device=device).float() 
        
        noise = torch.randn_like(images)
        noised_images = torch.zeros_like(images)
        predicted_noise = torch.zeros_like(images)
        t_tensor_norm = t / T

        for i in range(images.size(0)):
            t_single = t[i].unsqueeze(0)
            #t_norm = t_single / T
            schedule = self.scheduler(t_single) # Scheduler gets a int
            sqrt_alpha_bar_t = schedule['sqrt_alpha_bar']
            sqrt_one_minus_alpha_bar_t = schedule['sqrt_oneminus_alpha_bar']
            noised_image = sqrt_alpha_bar_t * images[i] + sqrt_one_minus_alpha_bar_t * noise[i]
            noised_images[i] = noised_image  # Assignment, not in-place
            #t_tensor = t_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3) # t_tensor is normalized
            #predicted_noise[i] = self.network(noised_image.unsqueeze(0), t_tensor, conditions_dropped[i].unsqueeze(0)) 
        predicted_noise = self.network(noised_images, t_tensor_norm, conditions_dropped) 
        noise_loss = self.loss_fn(predicted_noise, noise)

        return noise_loss








    def sample(self, conditions, omega):
        T = self.dmconfig.T
        
        # Step 1: Initialize x_T with noise from N(0, I)
        X_t = torch.randn((conditions.shape[0], 1, 28, 28), device=conditions.device)   
        # Convert conditions to one-hot encoding
        #conditions_onehot = F.one_hot(conditions, num_classes=10).float().to(conditions.device)

        with torch.no_grad():
            # Step 2-5: Iterate from t = T down to 1
            for t in reversed(range(1, T + 1)):
                # Create a tensor with a single element for current time step
                t_tensor = torch.full((1, 1, 1, 1), t, dtype=torch.float32, device=conditions.device)
                # norm before input to the network
                t_tensor_norm = t_tensor / T

                # Get scheduling parameters
                schedule = self.scheduler(t_tensor.view(1))  # Scheduler gets a scalar

                # Step 3: Sample z from N(0, I) if t > 1, else z = 0
                noise = torch.randn_like(X_t) if t > 1 else torch.zeros_like(X_t)     
                
                # Predict noise using the denoising network
                predicted_noise_conditional = self.network(X_t, t_tensor_norm.expand(conditions.shape[0], 1, 1, 1), conditions) 
                predicted_noise_nonconditional = self.network(X_t, t_tensor_norm.expand(conditions.shape[0], 1, 1, 1), torch.full_like(conditions, -1)) # include non-conditional term
                predicted_noise = (1+omega) * predicted_noise_conditional - omega * predicted_noise_nonconditional
                # Step 4: Update X_t using the provided formula
                one_minus_alpha_t = 1 - schedule['alpha_t']
                oneover_sqrt_alpha_t = schedule['oneover_sqrt_alpha']
                sqrt_one_minus_alpha_t_bar = schedule['sqrt_oneminus_alpha_bar']
                sqrt_beta_t = schedule['sqrt_beta_t']
                X_t = oneover_sqrt_alpha_t*(X_t - one_minus_alpha_t/sqrt_one_minus_alpha_t_bar * predicted_noise)  +  noise * sqrt_beta_t
        # Step 6: Return x_0, denormalize the output images
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images









# class ConditionalDDPM(nn.Module):
#     def __init__(self, dmconfig):
#         super().__init__()
#         self.dmconfig = dmconfig
#         self.loss_fn = nn.MSELoss()
#         self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

#     def scheduler(self, t_s):
#         beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
#         # ==================================================== #
#         # YOUR CODE HERE:
#         #   Inputs:
#         #       t_s: the input time steps, with shape (B,1). 
#         #   Outputs:
#         #       one dictionary containing the variance schedule
#         #       $\beta_t$ along with other potentially useful constants.       



#         # ==================================================== #
#         return {
#             'beta_t': beta_t,
#             'sqrt_beta_t': sqrt_beta_t,
#             'alpha_t': alpha_t,
#             'sqrt_alpha_bar': sqrt_alpha_bar,
#             'oneover_sqrt_alpha': oneover_sqrt_alpha,
#             'alpha_t_bar': alpha_t_bar,
#             'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
#         }

#     def forward(self, images, conditions):
#         T = self.dmconfig.T
#         noise_loss = None
#         # ==================================================== #
#         # YOUR CODE HERE:
#         #   Complete the training forward process based on the
#         #   given training algorithm.
#         #   Inputs:
#         #       images: real images from the dataset, with size (B,1,28,28).
#         #       conditions: condition labels, with size (B). You should
#         #                   convert it to one-hot encoded labels with size (B,10)
#         #                   before making it as the input of the denoising network.
#         #   Outputs:
#         #       noise_loss: loss computed by the self.loss_fn function  .  

#         pass



#         # ==================================================== #
        
#         return noise_loss

#     def sample(self, conditions, omega):
#         T = self.dmconfig.T
#         X_t = None
#         # ==================================================== #
#         # YOUR CODE HERE:
#         #   Complete the training forward process based on the
#         #   given sampling algorithm.
#         #   Inputs:
#         #       conditions: condition labels, with size (B). You should
#         #                   convert it to one-hot encoded labels with size (B,10)
#         #                   before making it as the input of the denoising network.
#         #       omega: conditional guidance weight.
#         #   Outputs:
#         #       generated_images  

#         pass



#         # ==================================================== #
#         generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
#         return generated_images



############################################################################################################
    # def scheduler(self, t_s):
    #     beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        
    #     # Generate a range of t values from 1 to t_s
    #     t_range = torch.arange(0, t_s , dtype=torch.float32) / (T - 1)
        
    #     # Linear interpolation to calculate beta_t at each value in t_range
    #     beta_t = beta_1 + (beta_T - beta_1) * t_range
        
    #     # Calculate corresponding alpha_t values
    #     alpha_t = 1 - beta_t
        
    #     # Calculate cumulative product of alpha_t up to t_s
        

    #     alpha_t_bar = torch.cumprod(alpha_t, dim=0)[-1]  # Take the last value as it represents the product up to t_s
    #     sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
    #     sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)
        
    #     # Return values
    #     return {
    #         'beta_t': beta_t[-1],  # Returning the beta value at t_s
    #         'sqrt_beta_t': torch.sqrt(beta_t[-1]),
    #         'alpha_t': alpha_t[-1],
    #         'sqrt_alpha_bar': sqrt_alpha_bar,
    #         'oneover_sqrt_alpha': 1 / torch.sqrt(alpha_t[-1]),
    #         'alpha_t_bar': alpha_t_bar,
    #         'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
    #     }