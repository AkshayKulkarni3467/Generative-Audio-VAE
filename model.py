import torch
from torch import nn
from dataloader import createDataLoader
import torch.nn.functional as F
from torch import nn,optim
from tqdm import tqdm


file_path = 'datasets/fsdd/spectrograms/'


class VariationalAutoEncoder(nn.Module):
    def __init__(self,latent_space_dim):
        self.latent_space_dim = latent_space_dim
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 2,kernel_size = (3,3),stride = 1,padding = 'same')
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 2, out_channels = 4,kernel_size = (5,5),stride = 2)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = (3,3),stride = 2)
        self.r3 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim = 0,end_dim =3)
        self.mu = nn.Linear(in_features = 27776,out_features =self.latent_space_dim)
        self.log_variance = nn.Linear(in_features = 27776,out_features = self.latent_space_dim)
        
        #decoder
        self.decoder_input = nn.Linear(in_features = self.latent_space_dim,out_features = 27776)
        self.noise1 = nn.Linear(in_features=self.latent_space_dim,out_features=1020)
        self.noise2 = nn.Linear(in_features=self.latent_space_dim,out_features = 256)
        self.convT1 = nn.ConvTranspose2d(in_channels = 8,out_channels=4,kernel_size = (3,3),stride = 2)
        self.r4 = nn.ReLU()
        self.convT2 = nn.ConvTranspose2d(in_channels = 4,out_channels = 2,kernel_size = (5,5), stride = 2)
        self.r5 = nn.ReLU()
        self.convT3 = nn.ConvTranspose2d(in_channels = 2,out_channels = 1,kernel_size = (3,3),stride = 1)
        self.r6 = nn.ReLU()
        
        self.log_variance_value = None
        self.mu_value = None
        self.reconstruction_loss_weight = 1000000
        
        
        
    def forward_encoder(self,x):
        x = torch.permute(x,[0,3,1,2])
        x = self.conv1(x)
        x = self.r1(x)
        x = self.conv2(x)
        x = self.r2(x)
        x = self.conv3(x)
        x = self.r3(x)
        y = self.flatten(x)
        x1 = self.mu(y)
        self.mu_value = x1
        x2 = self.log_variance(y)
        self.log_variance_value = x2
        dist = torch.empty(20).normal_(mean=0,std=1)
        epi = x1 + torch.exp(x2/2)*dist
        return epi
        # return x
    
    def forward_decoder(self,x1):
        x = self.decoder_input(x1)
        x = x.view(-1,8,62,14)
        x = self.convT1(x)
        x = self.r4(x)
        x = self.convT2(x)
        x = self.r5(x)
        x = self.convT3(x)
        x = self.r6(x)
        x = torch.permute(x,[0,2,3,1])
        y = self.noise1(x1)
        y = y.view(4,255,1,1)
        a = self.noise2(x1)
        a = a.view(4,1,64,1)
        z = torch.cat((x,y),dim=2)
        zbar = torch.cat((z,a),dim=1)
        return zbar
    
    def forward(self,x):
        x= self.forward_encoder(x)
        x = self.forward_decoder(x)
        return x,self.mu_value,self.log_variance_value
        
        
    def reconstruct(self,images):
        with torch.no_grad():
            latent_representations = self.forward_encoder(images)
            reconstructed_images = self.forward_decoder(latent_representations)
        return reconstructed_images,latent_representations
    
    def _calculate_combined_loss(self,y_target,y_pred,mu,log_variance):
        recons_loss = self._calculate_recons_loss(y_target,y_pred,mu,log_variance)
        kl_loss = self._calculate_kl_loss(y_target,y_pred,mu,log_variance)
        combined_loss = self.reconstruction_loss_weight*recons_loss+kl_loss
        return combined_loss
    
    def _calculate_recons_loss(self,y_target,y_pred,mu,log_variance):
        error = y_target - y_pred
        recons_loss = torch.mean(torch.square(error),axis=[1,2,3])
        return recons_loss
    
    def _calculate_kl_loss(self,y_target,y_pred,mu,log_variance):
        kl_loss = -0.5 * torch.sum(1+log_variance-torch.square(mu)-torch.exp(log_variance),axis=0)
        return kl_loss
    
    
    

        
def train(epochs):
        count = 0
        dataset,dataloader = createDataLoader(file_path=file_path)
        model = VariationalAutoEncoder(latent_space_dim=20)
        optimizer = optim.Adam(model.parameters(),lr = 3e-4)
        for epoch in range(epochs):
           for i,x in enumerate(dataloader):
               count+=1
               x_recons,mu,log_variance = model(x)
               loss = model._calculate_combined_loss(x,x_recons,mu,log_variance)
               loss = torch.mean(loss)
               print(loss)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               if count % 1000 == 0:
                   torch.save(model.state_dict(),'models/vae_20.pth')
        

def eval(images):
    with torch.no_grad():
        model = VariationalAutoEncoder(latent_space_dim=10)
        model.load_state_dict(torch.load('models/vae_20.pth'))
        latent_representations,recons_images = model.reconstruct(images)
        print(model._calculate_kl_loss(images,recons_images).detach().numpy())
        
    


if __name__ == "__main__":
    train(300)
    



